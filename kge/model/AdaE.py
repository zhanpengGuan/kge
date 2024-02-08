import time
import torch
from kge import Config, Dataset
from kge.model.kge_model import KgeEmbedder, RelationalScorer, KgeModel
# import nn for torch
import torch.nn as nn
from kge.model.reciprocal_relations_model import ReciprocalRelationsModel
from kge.util import  KgeLoss, KgeLRScheduler
DEVICE = 'cpu'
import torch.nn.functional as F
from kge.util import KgeSampler

from torch import Tensor
SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]

class AdaE(ReciprocalRelationsModel):
    r"""Implementation of the AdaE KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
    ):
        
        
      
 
        # self._init_configuration(config, configuration_key)
        # alt_dataset = dataset.shallow_copy()
        # base_model = KgeModel.create(
        #     config=config,
        #     dataset=alt_dataset,
        #     configuration_key=self.configuration_key + ".base_model",
        #     init_for_load_only=init_for_load_only,
        # )
        # # Initialize this model
        # super().__init__(
        #     config=config,
        #     dataset=dataset,
        #     scorer=base_model.get_scorer(),
        #     create_embedders=False,
        #     init_for_load_only=init_for_load_only,
        # )
        # self._base_model = base_model
        super().__init__(
        config=config,
        dataset=dataset,
        configuration_key=None,
        init_for_load_only=init_for_load_only,)


        # 
        # TODO change entity_embedder assignment to sub and obj embedders when support
        # for that is added
        if False:
            checkpoint = torch.load('/home/guanzp/code/AdaE/kge/local/wnrr/rank/20240111-132431AdaE_rank/checkpoint_best.pt')
            ent = checkpoint['model'][0]['_base_model._entity_embedder._embeddings.weight']
            rel = checkpoint['model'][0]['_base_model._relation_embedder._embeddings.weight']
            self._base_model.get_s_embedder()._embeddings.weight = nn.Parameter(ent, requires_grad=True)
            self._base_model.get_p_embedder()._embeddings.weight = nn.Parameter(rel, requires_grad=True)
        self._entity_embedder = self._base_model.get_s_embedder()
        
        self._relation_embedder = self._base_model.get_p_embedder()
        self.device = self.config.get("job.device")
        self._sampler = KgeSampler.create(config, "negative_sampling", dataset)
        self._max_subbatch_size: int = config.get("train.subbatch_size")
        self.loss = KgeLoss.create(config)
        self.is_forward_only = init_for_load_only
        self.type = config.options['AdaE_config']['type']


    def prepare_job(self, job, **kwargs):
        self._base_model.prepare_job(job, **kwargs)
    def _loss(self, batch_index, batch, is_arch = True):
        if  self.type =='ng_sample':
            return self._loss_ng(batch_index, batch,is_arch)
        elif self.type == '1vsall':
            return self._loss_1all(batch_index, batch,is_arch)
        elif self.type == 'kvsall':
            return self._loss_kall(batch_index, batch,is_arch)
        else:
            raise ValueError("no type {} in training ways".format(self.type))
        

    def _prepare_batch_ng(
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
    def _loss_ng(self, batch_index, batch, is_arch = True):
        "Breaks a batch into subbatches and processes them in turn."
       

        from kge.job.train import TrainingJob
        result = TrainingJob._ProcessBatchResult()
        self._prepare_batch_ng(batch_index, batch, result)
        batch_size = result.size

        max_subbatch_size = (
            self._max_subbatch_size if self._max_subbatch_size > 0 else batch_size
        )
        for subbatch_start in range(0, batch_size, max_subbatch_size):
            # determine data used for this subbatch
            subbatch_end = min(subbatch_start + max_subbatch_size, batch_size)
            subbatch_slice = slice(subbatch_start, subbatch_end)
            self._subloss_ng(batch_index, batch, subbatch_slice, result, is_arch)

        return result
    def _subloss_ng(self,
            batch_index,
            batch,
            subbatch_slice,
            result,
            is_arch
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
            loss_value_torch = self.loss(scores, labels[slot], num_negatives=num_samples) / batch_size
            
            result.avg_loss += loss_value_torch
            result.forward_time += time.time()

            # # backward pass for this slot in the subbatch
            # result.backward_time -= time.time()
            # if not self.is_forward_only:
            #     loss_value_torch.backward()
            # result.backward_time += time.time()
    
    def _loss_1all(self, batch_index, batch, is_arch = True):
        "Breaks a batch into subbatches and processes them in turn."
        from kge.job.train import TrainingJob
        result = TrainingJob._ProcessBatchResult()
        self._prepare_batch_1all(batch_index, batch, result)
        batch_size = result.size

        max_subbatch_size = (
            self._max_subbatch_size if self._max_subbatch_size > 0 else batch_size
        )
        for subbatch_start in range(0, batch_size, max_subbatch_size):
            # determine data used for this subbatch
            subbatch_end = min(subbatch_start + max_subbatch_size, batch_size)
            subbatch_slice = slice(subbatch_start, subbatch_end)
            self._subloss_1all(batch_index, batch, subbatch_slice, result)

        return result
    def _prepare_batch_1all(
            self, batch_index, batch, result
        ):
            result.size = len(batch["triples"])

    def _subloss_1all(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result
    ):
        batch_size = result.size

        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice].to(self.device)
        result.prepare_time += time.time()

        # forward/backward pass (sp)
        result.forward_time -= time.time()
        scores_sp = self.score_sp(triples[:, 0], triples[:, 1])
        loss_value_sp = self.loss(scores_sp, triples[:, 2]) / batch_size
        result.avg_loss += loss_value_sp
        result.forward_time += time.time()
        # result.backward_time = -time.time()
        # if not self.is_forward_only:
        #     loss_value_sp.backward()
        # result.backward_time += time.time()

        # forward/backward pass (po)
        result.forward_time -= time.time()
        scores_po = self.score_po(triples[:, 1], triples[:, 2])
        loss_value_po = self.loss(scores_po, triples[:, 0]) / batch_size
        result.avg_loss += loss_value_po
        result.forward_time += time.time()
        # result.backward_time -= time.time()
        # if not self.is_forward_only:
        #     loss_value_po.backward()
        # result.backward_time += time.time()

    def _prepare_batch_kall(
        self, batch_index, batch, result
    ):
        # move labels to GPU for entire batch (else somewhat costly, but this should be
        # reasonably small)
        result.prepare_time -= time.time()
        batch["label_coords"] = batch["label_coords"].to(self.device)
        result.size = len(batch["queries"])
        result.prepare_time += time.time()
    def _loss_kall(self, batch_index, batch, is_arch = True):
        "Breaks a batch into subbatches and processes them in turn."
        from kge.job.train import TrainingJob
        result = TrainingJob._ProcessBatchResult()
        self._prepare_batch_kall(batch_index, batch, result)
        batch_size = result.size

        max_subbatch_size = (
            self._max_subbatch_size if self._max_subbatch_size > 0 else batch_size
        )
        for subbatch_start in range(0, batch_size, max_subbatch_size):
            # determine data used for this subbatch
            subbatch_end = min(subbatch_start + max_subbatch_size, batch_size)
            subbatch_slice = slice(subbatch_start, subbatch_end)
            self._subloss_kall(batch_index, batch, subbatch_slice, result)

        return result
    def _subloss_kall(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result
    ):
        batch_size = result.size

        # prepare
        result.prepare_time -= time.time()
        queries_subbatch = batch["queries"][subbatch_slice].to(self.device)
        subbatch_size = len(queries_subbatch)
        label_coords_batch = batch["label_coords"]
        query_type_indexes_subbatch = batch["query_type_indexes"][subbatch_slice]

        # in this method, example refers to the index of an example in the batch, i.e.,
        # it takes values in 0,1,...,batch_size-1
        examples_for_query_type = {}
        for query_type_index, query_type in enumerate(self.query_types):
            examples_for_query_type[query_type] = (
                (query_type_indexes_subbatch == query_type_index)
                .nonzero(as_tuple=False)
                .to(self.device)
                .view(-1)
            )

        labels_subbatch = kge.job.util.coord_to_sparse_tensor(
            subbatch_size,
            max(self.dataset.num_entities(), self.dataset.num_relations()),
            label_coords_batch,
            self.device,
            row_slice=subbatch_slice,
        ).to_dense()
        labels_for_query_type = {}
        for query_type, examples in examples_for_query_type.items():
            if query_type == "s_o":
                labels_for_query_type[query_type] = labels_subbatch[
                    examples, : self.dataset.num_relations()
                ]
            else:
                labels_for_query_type[query_type] = labels_subbatch[
                    examples, : self.dataset.num_entities()
                ]

        if self.label_smoothing > 0.0:
            # as in ConvE: https://github.com/TimDettmers/ConvE
            for query_type, labels in labels_for_query_type.items():
                if query_type != "s_o":  # entity targets only for now
                    labels_for_query_type[query_type] = (
                        1.0 - self.label_smoothing
                    ) * labels + 1.0 / labels.size(1)

        result.prepare_time += time.time()

        # forward/backward pass (sp)
        for query_type, examples in examples_for_query_type.items():
            if len(examples) > 0:
                result.forward_time -= time.time()
                if query_type == "sp_":
                    scores = self.score_sp(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                elif query_type == "s_o":
                    scores = self.score_so(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                else:
                    scores = self.model.score_po(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                # note: average on batch_size, not on subbatch_size
                loss_value = (
                    self.loss(scores, labels_for_query_type[query_type]) / batch_size
                )
                result.avg_loss += loss_value.item()
                result.forward_time += time.time()
                # result.backward_time -= time.time()
                # if not self.is_forward_only:
                #     loss_value.backward()
                # result.backward_time += time.time()
