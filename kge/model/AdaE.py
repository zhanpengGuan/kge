import time
import torch
from kge import Config, Dataset
from kge.model.kge_model import KgeEmbedder, RelationalScorer, KgeModel
# import nn for torch
import torch.nn as nn

from kge.util import  KgeOptimizer, KgeLRScheduler
DEVICE = 'cpu'
import torch.nn.functional as F
from kge.util import KgeSampler
SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]

class AdaEScorer(RelationalScorer):
    r"""Implementation of the AdaE KGE scorer.

    Reference: Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier and
    Guillaume Bouchard: AdaE Embeddings for Simple Link Prediction. ICML 2016.
    `<http://proceedings.mlr.press/v48/trouillon16.pdf>`_

    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        # Here we use a fast implementation of computing the AdaE scores using
        # Hadamard products, as in Eq. (11) of paper.
        #
        # Split the relation and object embeddings into real part (first half) and
        # imaginary part (second half).
        p_emb_re, p_emb_im = (t.contiguous() for t in p_emb.chunk(2, dim=1))
        o_emb_re, o_emb_im = (t.contiguous() for t in o_emb.chunk(2, dim=1))

        # combine them again to create a column block for each required combination
        s_all = torch.cat((s_emb, s_emb), dim=1)  # re, im, re, im
        r_all = torch.cat((p_emb_re, p_emb, -p_emb_im), dim=1)  # re, re, im, -im
        o_all = torch.cat((o_emb, o_emb_im, o_emb_re), dim=1)  # re, im, im, re

        if combine == "spo":
            out = (s_all * o_all * r_all).sum(dim=1)
        elif combine == "sp_":
            out = (s_all * r_all).mm(o_all.transpose(0, 1))
        elif combine == "_po":
            out = (r_all * o_all).mm(s_all.transpose(0, 1))
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class AdaE(KgeModel):
    r"""Implementation of the AdaE KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=AdaEScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        self.device = self.config.get("job.device")
        self._sampler = KgeSampler.create(config, "negative_sampling", dataset)
        self._max_subbatch_size: int = config.get("train.subbatch_size")
    # def get_s_embedder(self) -> KgeEmbedder:
    #         return self._multi_entity_embedder
    # def get_o_embedder(self) -> KgeEmbedder:
    #         return self._multi_entity_embedder
    # def get_p_embedder(self) -> KgeEmbedder:
    #         return self._multi_relation_embedder

    def _prepare_batch(
        self, batch_index, batch, result
    ):
        # move triples and negatives to GPU. With some implementaiton effort, this may
        # be avoided.
        result.prepare_time -= time.time()
        batch["triples"] = batch["triples"].to(self.device)
        for ns in batch["negative_samples"]:
            ns.positive_triples = batch["triples"]
        batch["negative_samples"] = [
            ns.to(self.device) for ns in batch["negative_samples"]
        ]

        batch["labels"] = [None] * 3  # reuse label tensors b/w subbatches
        result.size = len(batch["triples"])
        result.prepare_time += time.time()

    def _loss(self, batch_index, batch):
        "Breaks a batch into subbatches and processes them in turn."
        from kge.job.train import TrainingJob
        result = TrainingJob._ProcessBatchResult()
        self._prepare_batch(batch_index, batch, result)
        batch_size = result.size

        max_subbatch_size = (
            self._max_subbatch_size if self._max_subbatch_size > 0 else batch_size
        )
        for subbatch_start in range(0, batch_size, max_subbatch_size):
            # determine data used for this subbatch
            subbatch_end = min(subbatch_start + max_subbatch_size, batch_size)
            subbatch_slice = slice(subbatch_start, subbatch_end)
            self._subloss(batch_index, batch, subbatch_slice, result)

        return result
    def _subloss(self,
            batch_index,
            batch,
            subbatch_slice,
            result,
            ):
        batch_size = result.size

        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice]
        batch_negative_samples = batch["negative_samples"]
        subbatch_size = len(triples)
        result.prepare_time += time.time()
        labels = batch["labels"]  # reuse b/w subbatches

        # process the subbatch for each slot separately
        for slot in [S, P, O]:
            num_samples = self._sampler.num_samples[slot]
            if num_samples <= 0:
                continue

            # construct gold labels: first column corresponds to positives,
            # remaining columns to negatives
            if labels[slot] is None or labels[slot].shape != (
                subbatch_size,
                1 + num_samples,
            ):
                result.prepare_time -= time.time()
                labels[slot] = torch.zeros(
                    (subbatch_size, 1 + num_samples), device=self.device
                )
                labels[slot][:, 0] = 1
                result.prepare_time += time.time()

            # compute the scores
            result.forward_time -= time.time()
            scores = torch.empty((subbatch_size, num_samples + 1), device=self.device)
            scores[:, 0] = self.score_spo(
                triples[:, S], triples[:, P], triples[:, O], direction=SLOT_STR[slot],
            )
            result.forward_time += time.time()
            scores[:, 1:] = batch_negative_samples[slot].score(
                self, indexes=subbatch_slice
            )
            result.forward_time += batch_negative_samples[slot].forward_time
            result.prepare_time += batch_negative_samples[slot].prepare_time

            # compute loss for slot in subbatch (concluding the forward pass)
            result.forward_time -= time.time()
            loss_value_torch = (
                self.loss(scores, labels[slot], num_negatives=num_samples) / batch_size
            )
            result.avg_loss += loss_value_torch.item()
            result.forward_time += time.time()

            # backward pass for this slot in the subbatch
            result.backward_time -= time.time()
            if not self.is_forward_only:
                loss_value_torch.backward()
            result.backward_time += time.time()
    def score_spo(self, s, p, o, direction=None):
        if direction == "o":
            return super().score_spo(s, p, o, "o")
        elif direction == "s":
            return super().score_spo(o, p + self.dataset.num_relations()/2, s, "o")
        else:
            raise Exception(
                "The reciprocal relations model cannot compute "
                "undirected spo scores."
            )