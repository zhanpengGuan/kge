from torch import Tensor
import torch.nn as nn
import torch.nn.functional

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List

DEVICE = "cpu"
class Multi_LookupEmbedder(KgeEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        init_for_load_only=False,
    ):
        super().__init__(
            config, dataset, configuration_key, init_for_load_only=init_for_load_only
        )

        # read config
        self.normalize_p = self.get_option("normalize.p")
        self.space = self.check_option("space", ["euclidean", "complex"])

        # n3 is only accepted when space is complex
        if self.space == "complex":
            self.regularize = self.check_option("regularize", ["", "lp", "n3"])
        else:
            self.regularize = self.check_option("regularize", ["", "lp"])

        self.sparse = self.get_option("sparse")
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size

        round_embedder_dim_to = self.get_option("round_dim_to")
        if len(round_embedder_dim_to) > 0:
            self.dim = round_to_points(round_embedder_dim_to, self.dim)

        # adaE init
        self.adaE_init()


        if not init_for_load_only:
            # initialize weights
            self.initialize(self._embeddings.weight.data)
            self._normalize_embeddings()

        # TODO handling negative dropout because using it with ax searches for now
        dropout = self.get_option("dropout")
        if dropout < 0:
            if config.get("job.auto_correct"):
                config.log(
                    "Setting {}.dropout to 0., "
                    "was set to {}.".format(configuration_key, dropout)
                )
                dropout = 0
        self.dropout = torch.nn.Dropout(dropout)

    def _normalize_embeddings(self):
        if self.normalize_p > 0:
            with torch.no_grad():
                self._embeddings.weight.data = torch.nn.functional.normalize(
                    self._embeddings.weight.data, p=self.normalize_p, dim=-1
                )

    def prepare_job(self, job: Job, **kwargs):
        from kge.job import TrainingJob

        super().prepare_job(job, **kwargs)
        if self.normalize_p > 0 and isinstance(job, TrainingJob):
            # just to be sure it's right initially
            job.pre_run_hooks.append(lambda job: self._normalize_embeddings())

            # normalize after each batch
            job.post_batch_hooks.append(lambda job: self._normalize_embeddings())

    @torch.no_grad()
    def init_pretrained(self, pretrained_embedder: KgeEmbedder) -> None:
        (
            self_intersect_ind,
            pretrained_intersect_ind,
        ) = self._intersect_ids_with_pretrained_embedder(pretrained_embedder)
        self._embeddings.weight[
            torch.from_numpy(self_intersect_ind)
            .to(self._embeddings.weight.device)
            .long()
        ] = pretrained_embedder.embed(torch.from_numpy(pretrained_intersect_ind)).to(
            self._embeddings.weight.device
        )

    def embed(self, indexes: Tensor) -> Tensor:
        return self._postprocess(self._adaE(indexes = indexes.long()))

    def embed_all(self) -> Tensor:
        # undo
        return self._postprocess(self._embeddings_all())

    def _postprocess(self, embeddings: Tensor) -> Tensor:
        
        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)
        return embeddings

    def _embeddings_all(self) -> Tensor:
        return self._embeddings(
            torch.arange(
                self.vocab_size, dtype=torch.long, device=self._embeddings.weight.device
            )
        )

    def _get_regularize_weight(self) -> Tensor:
        return self.get_option("regularize_weight")

    def _abs_complex(self, parameters) -> Tensor:
        parameters_re, parameters_im = (t.contiguous() for t in parameters.chunk(2, dim=1))
        parameters = torch.sqrt(parameters_re ** 2 + parameters_im ** 2 + 1e-14) # + 1e-14 to avoid NaN: https://github.com/lilanxiao/Rotated_IoU/issues/20
        return parameters

    def penalty(self, **kwargs) -> List[Tensor]:
        # TODO factor out to a utility method
        result = super().penalty(**kwargs)
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            pass
        elif self.regularize in ["lp", 'n3']:
            if self.regularize == "n3":
                p = 3
            else:
                p = (
                    self.get_option("regularize_args.p")
                    if self.has_option("regularize_args.p")
                    else 2
                )
            regularize_weight = self._get_regularize_weight()
            if not self.get_option("regularize_args.weighted"):
                # unweighted Lp regularization
                parameters = self._embeddings_all()
                if self.regularize == "n3" and self.space == 'complex':
                    parameters = self._abs_complex(parameters)
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (regularize_weight / p * parameters.norm(p=p) ** p).sum(),
                    )
                ]
            else:
                # weighted Lp regularization
                unique_indexes, counts = torch.unique(
                    kwargs["indexes"], return_counts=True
                )
                parameters = self._embeddings(unique_indexes)

                if self.regularize == "n3" and self.space == 'complex':
                    parameters = self._abs_complex(parameters)

                if (p % 2 == 1) and (self.regularize != "n3"):
                    parameters = torch.abs(parameters)
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (
                            regularize_weight
                            / p
                            * (parameters ** p * counts.float().view(-1, 1))
                        ).sum()
                        # In contrast to unweighted Lp regularization, rescaling by
                        # number of triples/indexes is necessary here so that penalty
                        # term is correct in expectation
                        / len(kwargs["indexes"]),
                    )
                ]
        else:  # unknown regularization
            raise ValueError(f"Invalid value regularize={self.regularize}")

        return result
    def adaE_init(self):
        
        self.adae_config = self.config.options['AdaE_config']
        
        if self.adae_config['train_mode'] in ['original','fix']:
            # if train_mode is original or fix, the embeddings must be Embedding class
            self._embeddings = torch.nn.Embedding(
                self.vocab_size, self.dim, sparse=self.sparse,
            )
            # if train_mode is fix, the embedder must have Transform_layer class
            if self.adae_config['train_mode'] == 'fix':
                self.Transform_layer = nn.Linear(self.dim, self.dim)
        else:
            self._embeddings = torch.zeros( self.vocab_size, self.dim).to(DEVICE)
            self.dim_list = self.config.options['AdaE_config']['emb_list']
            self._embeddings_list =nn.ParameterList( [nn.Parameter(torch.zeros(self.vocab_size, i)) for i in self.dim_list])

       
    def _adaE(self, **kwargs) -> KgeEmbedder:
      
        kwargs['all'] = kwargs.get('all', False)
        indexes = kwargs.get('indexes', None)
        if self.adae_config['train_mode'] in ['original','fix', 'rank','auto']:
            if self.adae_config['train_mode'] == 'original':
                # original means no change
                emb = self._embeddings(indexes)
            if self.adae_config['train_mode'] == 'fix':
                # fix means all dim is same
                emb = self._embeddings(indexes)
                emb = self.Transform_layer(emb) 
            elif self.adae_config['train_mode'] == 'rank':
                # rank means use ranked dim with each entity 
                pro = self.picker_rank(indexes,negs=None)
                emb = self.aligment_fix(indexes,probability=pro)
            elif self.adae_config['train_mode'] == 'share_rank':
                pass
            elif self.adae_config['train_mode'] == 'share_rank_zp':
                pass
            elif self.adae_config['train_mode'] == 'auto':
                pass
        return emb
   
    def picker_rank(self, triples):
        """
        using pre frequency as choice of entity embedding size despite r
        """          
        batch_size=len(triples[:,0])
        # 上次的emb，离散选择
        label_h  =  self.rank_e[triples[:, 0]].unsqueeze(-1)
        pro = torch.zeros(batch_size,self.dim_list_size).to(DEVICE).scatter_(1, label_h, 1)       
        
        return pro.detach()
    def aligment_fix(self, triples, negs=None, mode="single" , probability=None,step = 10000):
        """
        fixed ,which means no gumbel softmax
        """
        l2_norm = 1
        if mode == "single":
            head_emb = []
            for i in range(0,self.dim_list_size):     
                head_emb.append(self.ent_emb_list[i][triples[:, 0]])  # [bs, 1, dim]
            output_h = []
            for i in range(0,self.dim_list_size):       
                output_h_pre = (self.Selection(self.Transform_e_layer_list[i](head_emb[i]))*l2_norm)
                output_h.append(output_h_pre)
            # 堆叠以便于计算
            head_s = torch.stack(output_h, dim=1)
            # no Gumbal softmax probability
            Gpro_h = probability[0]
            #sot selection   
            head_final = torch.mul(head_s, Gpro_h.unsqueeze(-1)).sum(dim = 1).unsqueeze(1)
            with torch.no_grad():
                # save fianl emb
                self.entity_embedding[triples[:, 0]] = head_final.squeeze(1)
                # save pro
                self.ent_choice_emb[triples[:, 0]] = probability[0]
        return head_final