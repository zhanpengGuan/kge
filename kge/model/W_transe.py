import torch
from torch import Tensor
from kge import Config, Dataset
from kge.job import Job
from kge.model.kge_model import RelationalScorer, KgeModel,KgeEmbedder
from torch.nn import functional as F
from copy import deepcopy

class W_TransEScorer(RelationalScorer):
    r"""Implementation of the TransE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")
        self._BP = False
        
        

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)
        # compute theta(s_emb,p_emb,o_emb) with Adam Optimization process

        if combine == "spo":
            out = -F.pairwise_distance(s_emb + p_emb, o_emb, p=self._norm)
        elif combine == "sp_":
            # we do not use matrix multiplication due to this issue
            # https://github.com/pytorch/pytorch/issues/42479
            out = -torch.cdist(
                s_emb + p_emb,
                o_emb,
                p=self._norm,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
        elif combine == "_po":
            out = -torch.cdist(
                o_emb - p_emb,
                s_emb,
                p=self._norm,
                compute_mode="donot_use_mm_for_euclid_dist",
            )
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)
        return out.view(n, -1)
    def get_next_emb_adam(grad,s_emb,p_emb,o_emb,curr_optim_status):
        
        return s_emb,p_emb,o_emb


class W_TransE(KgeModel):
    r"""Implementation of the TransE KGE model."""

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
            scorer=W_TransEScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
        )
        #clip  the weight
        self.clip = config.get("W_transe")["clip"]
        self.W_update = False
        self.outer = False
        self.g_r = config.get("W_transe")["golden_ratio"]
        self.init = config.get("W_transe")["init"]
        self.require_grad_relation = config.get("W_transe")["require_grad_relation"]
        self.require_grad_entity = config.get("W_transe")["require_grad_entity"]
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
        self._weight_entity.requires_grad = config.get("W_transe")["require_grad_entity"]
        self._weight_relation = torch.nn.Parameter(init_relation*torch.ones(dataset.num_relations()))
        self._weight_relation.requires_grad = False
        # assumed_emb
        self.assumed_emb = None

    def prepare_job(self, job: Job, **kwargs):
        super().prepare_job(job, **kwargs)

        from kge.job import TrainingJobNegativeSampling

        if (
            isinstance(job, TrainingJobNegativeSampling)
            and job.config.get("negative_sampling.implementation") == "auto"
        ):
            # TransE with batch currently tends to run out of memory, so we use triple.
            job.config.set("negative_sampling.implementation", "triple", log=True)
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