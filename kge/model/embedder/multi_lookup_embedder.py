import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points

from typing import List


DEVICE = 'cuda'
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
        self.adaE_init(dataset, init_for_load_only=init_for_load_only)
        

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
    def _normalize_embeddings_list(self):
        if self.normalize_p > 0:
            with torch.no_grad():
                for i in range(len(self.dim_list)):
                    self._embeddings_list[i].weight.data = torch.nn.functional.normalize(
                        self._embeddings_list.weight.data, p=self.normalize_p, dim=-1
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
                        # number of indexes/indexes is necessary here so that penalty
                        # term is correct in expectation
                        / len(kwargs["indexes"]),
                    )
                ]
        else:  # unknown regularization
            raise ValueError(f"Invalid value regularize={self.regularize}")

        return result
    def adaE_init(self, dataset, init_for_load_only=False):
        self.adae_config = self.config.options['AdaE_config']
        self.device = self.config.get("job.device")
        # dataset
        # 数字越大频率越高
        self.rank_e, self.rank_r = dataset.count_entity_frequency(dataset._triples['train'], dataset._num_entities, dataset._num_relations, self.adae_config['choice_list'] )
        self.rank_e, self.rank_r = self.rank_e.to(self.device), self.rank_r.to(self.device)
        self.dim_list = self.config.options['AdaE_config']['dim_list']
        # if train_mode is original or fix, the embeddings must be Embedding class
        self._embeddings = torch.nn.Embedding(
            self.vocab_size, self.dim, sparse=self.sparse,
        )
        if not init_for_load_only:
            # initialize weights
            self.initialize(self._embeddings.weight.data)
            self._normalize_embeddings()

        if self.adae_config['train_mode'] in ['original','fix']:
            # if train_mode is fix, the embedder must have Transform_layer class
            if self.adae_config['train_mode'] == 'fix':
                self.Transform_layer = nn.Linear(self.dim, self.dim)
                nn.init.xavier_uniform_(self.Transform_layer.weight.data)
        elif self.adae_config['train_mode'] in ['rank','auto']:
            if self.adae_config['train_mode'] == 'auto':
                self.picker = Picker(self.config, dataset)
            self.t_s = self.adae_config['t_s']*2
            self.choice_emb = torch.zeros(self.vocab_size, len(self.dim_list)).to(DEVICE)
            nn.init.uniform_(tensor=self.choice_emb)
            self.BN = nn.LayerNorm(self.dim).to(DEVICE)
            self.AF = nn.Tanh()
            self.Selection = nn.Sequential(
                    self.BN,
                    self.AF,
                    nn.Dropout(0.2),
                    )
            for i in range(0,len(self.dim_list)):
                self.Transform_layer_list =nn.ModuleList([nn.Linear(self.dim_list[i],self.dim) for i in range(0,len(self.dim_list))])
                nn.init.xavier_uniform_(self.Transform_layer_list[i].weight.data)
            
            self._embeddings_list =nn.ParameterList( [nn.Parameter(torch.zeros(self.vocab_size, i)) for i in self.dim_list])
          
            if not init_for_load_only:
                for i in range(len(self.dim_list)):
                    self.initialize(self._embeddings_list[i])
                    self._normalize_embeddings_list()
        else:
            raise ValueError(f"Invalid value train_mode={self.adae_config['train_mode']}")
       
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
                pro = self._picker_rank(indexes)
                emb = self._aligment_fix(indexes,probability=pro)
            elif self.adae_config['train_mode'] == 'share_rank':
                pass
            elif self.adae_config['train_mode'] == 'share_rank_zp':
                pass
            elif self.adae_config['train_mode'] == 'auto':
                # rank means use ranked dim with each entity 
                pro = self._picker(indexes)
                emb = self._aligment(indexes,probability=pro)

        return emb
   
    def _picker_rank(self, indexes):
        """
        using pre frequency as choice of entity embedding size despite r
        """          
        batch_size=len(indexes)
        # 上次的emb，离散选择
        label  =  self.rank_e[indexes].unsqueeze(-1)
        pro = torch.zeros(batch_size,len(self.dim_list)).to(DEVICE).scatter_(1, label, 1)    

        return pro.detach()
    
    def _picker(self, indexes):
        """
        picker  process
        """          
        input_h =  torch.cat((self._embeddings(indexes), self.picker.bucket(self.rank_e[indexes])),dim = 1)
        pro = F.softmax(self.picker.forward(input_h),dim=-1)

        return pro.detach()

    def _aligment_fix(self, indexes, probability=None,step = 10000):
        """
        fixed ,which means no gumbel softmax
        """
        l2_norm = 1
    
        head_emb = []
        for i in range(0,len(self.dim_list)):     
            head_emb.append(self._embeddings_list[i][indexes])  # [bs, 1, dim]
        output_h = []
        for i in range(0,len(self.dim_list)):       
            output_h_pre = (self.Selection(self.Transform_layer_list[i](head_emb[i]))*l2_norm)
            output_h.append(output_h_pre)
        # 堆叠以便于计算
        head_s = torch.stack(output_h, dim=1)
        # no Gumbal softmax probability
        Gpro_h = probability
        #soft selection   
        head_final = torch.mul(head_s, Gpro_h.unsqueeze(-1)).sum(dim = 1)
        with torch.no_grad():
            # save fianl emb
            self._embeddings.weight.data[indexes] = head_final
            # save pro
            self.choice_emb[indexes] = probability

        return head_final
    
    def _aligment(self, indexes,  probability=None,step = 10000):
        """
        align with gumbal softmax
        """
        l2_norm = 1
        Tau=max(0.01,1-5.0e-5*step)
        head_emb = []
        for i in range(0,len(self.dim_list)):     
            head_emb.append(self._embeddings_list[i][indexes])  # [bs, 1, dim]
        output_h = []
        for i in range(0,len(self.dim_list)):       
            output_h_pre = (self.Selection(self.Transform_layer_list[i](head_emb[i]))*l2_norm)
            output_h.append(output_h_pre)
        # 堆叠以便于计算
        head_s = torch.stack(output_h, dim=1)
        # Gumbal softmax probability
        Gpro_h = F.gumbel_softmax(probability, tau=Tau, hard=True)
        #soft selection   
        head_final = torch.mul(head_s, Gpro_h.unsqueeze(-1)).sum(dim = 1)
        with torch.no_grad():
            # save fianl emb
            self._embeddings.weight.data[indexes] = head_final
            # save pro
            self.choice_emb[indexes] = probability

        return head_final

class Picker:
    def __init__( self, config: Config, dataset: Dataset) -> None:
        # # father init
        # super(Picker, self).__init__()
        self.adae_config = config.options['AdaE_config']
        self.device = config.get("job.device")
        self.dim: int = config.options['multi_lookup_embedder']['dim']
        self.dim_list_size = len(self.adae_config['dim_list'])
        self.dim_bucket = int(self.adae_config['t_s']/8)
        
        self.FC1 = nn.Linear(self.dim_bucket+self.dim,128).to(self.device)
        self.FC2 = nn.Linear(128,64).to(self.device)
        self.FC3 = nn.Linear(64,self.dim_list_size).to(self.device)
        nn.init.xavier_uniform_(self.FC1.weight.data)
        nn.init.xavier_uniform_(self.FC2.weight.data)
        nn.init.xavier_uniform_(self.FC3.weight.data)  
        self.Picker = nn.Sequential(
            self.FC1,
            nn.Dropout(0.5),
            nn.Tanh(),
            self.FC2,
            nn.Dropout(0.5),
            nn.Tanh(),
            self.FC3
        ) 
        # self.FC1_r = nn.Linear(self.dim_bucket+self.dim,128).to(DEVICE)
        # self.FC2_r = nn.Linear(128,64).to(DEVICE)
        # self.FC3_r = nn.Linear(64,self.dim_list_size).to(DEVICE)
        # nn.init.xavier_uniform_(self.FC1_r.weight.data)
        # nn.init.xavier_uniform_(self.FC2_r.weight.data)
        # nn.init.xavier_uniform_(self.FC3_r.weight.data)
        # self.Picker_r = nn.Sequential(
        #     self.FC1_r,
        #     nn.Dropout(0.5),
        #     nn.Tanh(),
        #     self.FC2_r,
        #     nn.Dropout(0.5),
        #     nn.Tanh(),
        #     self.FC3_r
        #     )

        # bucket emb
        self.k = self.dim_list_size
        self.bucket = nn.Embedding(self.k, self.dim_bucket).to(self.device)
    #
    def forward(self, input):
        # 上次的emb，离散选择
        pro = F.softmax(self.Picker(input),dim=-1)

        return pro