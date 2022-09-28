import torch
from torch import Tensor
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from copy import deepcopy

class W_ComplExScorer(RelationalScorer):
    r"""Implementation of the ComplEx KGE scorer.

    Reference: Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier and
    Guillaume Bouchard: Complex Embeddings for Simple Link Prediction. ICML 2016.
    `<http://proceedings.mlr.press/v48/trouillon16.pdf>`_

    """

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)

        # Here we use a fast implementation of computing the ComplEx scores using
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


class W_ComplEx(KgeModel):
    r"""Implementation of the ComplEx KGE model."""

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
            scorer=W_ComplExScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        #clip  the weight
        self.clip = config.get("W_complex")["clip"]
        self.W_update = False
        self.outer = False
        self.g_r = config.get("W_complex")["golden_ratio"]
        self.init = config.get("W_complex")["init"]
        self.require_grad_relation = config.get("W_complex")["require_grad_relation"]
        self.require_grad_entity = config.get("W_complex")["require_grad_entity"]
        self.e_frq = self.get_e_init(dataset)
        self.r_frq = self.get_r_init(dataset)
        # compute frequency of entities/realtions
        if self.init:
            init_entity = self.get_e_init(dataset)
            init_relation = self.get_r_init(dataset)
        else:
            init_entity = torch.ones(dataset.num_entities())
            init_relation = torch.ones(dataset.num_relations())
        self._weight_entity = torch.nn.Parameter(init_entity*torch.ones(dataset.num_entities()))
        self._weight_entity.requires_grad = config.get("W_complex")["require_grad_entity"]
        self._weight_relation = torch.nn.Parameter(init_relation*torch.ones(dataset.num_relations()))
        self._weight_relation.requires_grad = False
        # assumed_emb
        self.assumed_emb = None
    def get_e_w(self,indexes:Tensor)-> Tensor:
        if hasattr(self,'W_update'):
            if self.W_update == True:
                return self._weight_entity[indexes.long()]
        return self._weight_entity[indexes.long()].data.detach()
    def get_r_w(self,indexes:Tensor)-> Tensor:
        if hasattr(self,'W_update'):
            if self.W_update == True:
                return self._weight_relation[indexes.long()]
        return self._weight_relation[indexes.long()].data.detach()

    def get_e_init(self,dataset)->Tensor:
        ret = torch.ones(dataset.num_entities())
        for i in dataset._triples['train']:
            ret[i[0]] += 1
            ret[i[2]] += 1
        golden_f = deepcopy(sorted(ret,reverse=True)[int(len(ret)*self.g_r)])
        for i in range(0,len(ret)):
            ret[i] = golden_f/ret[i] if golden_f/ret[i]<self.clip else self.clip
            

        return ret
    def get_r_init(self,dataset)->Tensor:
        ret = torch.ones(dataset.num_relations())
        for i in dataset._triples['train']:
            ret[i[1]] += 1
        golden_f = deepcopy(sorted(ret,reverse=True)[int(len(ret)*self.g_r)])
        for i in range(0,len(ret)):
            ret[i] = golden_f/ret[i] if golden_f/ret[i]<self.clip else self.clip
        return ret