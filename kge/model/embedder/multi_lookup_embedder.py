from asyncio import sleep
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from kge import Config, Dataset
from kge.job import Job
from kge.model import KgeEmbedder
from kge.misc import round_to_points
import numpy as np
from typing import List

import random
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
        self.configuration_key = configuration_key
        # n3 is only accepted when space is complex
        if self.space == "complex":
            self.regularize = self.check_option("regularize", ["", "lp", "n3"])
        else:
            self.regularize = self.check_option("regularize", ["", "lp"])
        DEVICE =  config.get("job.device")
        
        self.sparse = self.get_option("sparse")
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.vocab_size = vocab_size
        self.step = 0
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
                        self._embeddings_list[i].weight.data, p=self.normalize_p, dim=-1
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
        self.step += 1/3
        return self._postprocess(self._adaE(indexes = indexes.long(), training = self.training, step = self.step))
        # return self._postprocess(self._embeddings(indexes.long()))
    def embed_all(self) -> Tensor:
        self.step += 1/3
        # all indexes
        indexes = torch.arange(
                self.vocab_size, dtype=torch.long, device=self._embeddings.weight.device
            )
        return self._postprocess(self._adaE(indexes = indexes, training = self.training, step = self.step))

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

    def penalty_m(self) -> List[Tensor]:
        # TODO factor out to a utility method
       
        Tau=max(0.01,1*np.exp(-0.0003*self.step))
        pro = self._picker(self.all_indexes)
        Gpro = self.ste(pro, tau=Tau, hard=False)
        
        if self.adae_config['cie']:
            result = 0.3*(((self.dim_l*Gpro).sum()/(Gpro.shape[0]*max(self.dim_list))))**2
        else:
            result = 1*(((self.dim_l*Gpro).sum()/(Gpro.shape[0]*max(self.dim_list)))) 
        return result
    def ste(self, logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
  
        gumbels = (logits) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = y_hard - (y_soft).detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret

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
                indexes = torch.arange(
                    self.vocab_size, dtype=torch.long, device=self._embeddings.weight.device
                )
                parameters = self._adaE(indexes = indexes, training = self.training, step = self.step)
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
                
                parameters = self._adaE(indexes = unique_indexes, training = self.training, step = self.step)
               

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
        self.DP = nn.Dropout(0.2)
        # 是否在bi_level的过程中：意味着不需要对embeddings进行赋值
        self.is_bilevel = False
        # dataset
        # 数字越大频率越高
        self.rank_e, self.rank_r, self.count_e, self.pivot = dataset.count_entity_frequency(dataset._triples['train'], dataset._num_entities, dataset._num_relations, self.adae_config['choice_list'], self.dataset.folder)
        self.rank_e, self.rank_r = self.rank_e.to(self.device), self.rank_r.to(self.device)
        
        #retrain
        self.choice = None
        # print(self.adae_config['folder'])
        if  self.adae_config['folder']!="None":
            folder = self.adae_config['folder']
            probability = torch.load("local/experiments/fb15k-237/"+folder+"/checkpoint_best.pt")['choice_emb'].to(self.device)
            Gpro =  torch.zeros(probability.shape,device=self.device)
            Gpro_index = torch.argmax(probability, dim = -1).unsqueeze(-1)
            self.choice = Gpro.scatter_(1, Gpro_index, 1)
        
            


        if self.configuration_key.split('.')[-1]=="entity_embedder":
            self.rank = self.rank_e
            # self.rank = torch.tensor([i for i  in range(dataset._num_entities)],device = self.device)
        else:
            self.rank = self.rank_r
        self.dim_list = self.config.options['AdaE_config']['dim_list']
        # if train_mode is original or fix, the embeddings must be Embedding class
        self._embeddings = torch.nn.Embedding(
            self.vocab_size, self.dim, sparse=self.sparse,
        )
        if not init_for_load_only:
            # initialize weights
            self.initialize(self._embeddings.weight.data)
            self._normalize_embeddings()
      
            
        train_mode = self.adae_config['train_mode']
        if train_mode in ['fix','rank','auto']:
            self.AF = nn.Tanh().to(self.device)
            # cie continuous input embeddings
            if self.adae_config['cie']:
                assert self.adae_config['ali_way'] == 'zp'
            if self.adae_config['ali_way'] == 'ts':
                self.BN = nn.LayerNorm(self.dim).to(self.device)
                # self.BN = nn.BatchNorm1d(self.dim).to(self.device)
                self.Selection = nn.Sequential(
                    self.BN
                    )
            elif self.adae_config['ali_way'] == 'zp':
                # self.BN = nn.BatchNorm1d(self.dim).to(self.device)
                # self.BN = nn.LayerNorm(self.dim).to(self.device)
                # self.Selection = nn.Sequential(
                #     self.BN
                #     ) 
                pass
                

        if train_mode in ['original','fix']:
            # if train_mode is fix, the embedder must have Transform_layer class
            if self.adae_config['train_mode'] == 'fix':
                self.Transform_layer = nn.Linear(self.dim, self.dim)
                self.Transform_layer1 = nn.Linear(self.dim, self.dim)
                nn.init.xavier_uniform_(self.Transform_layer.weight.data)
                nn.init.xavier_uniform_(self.Transform_layer1.weight.data)
                # self.Transform_layer.weight.data = torch.eye(self.dim, self.dim)
        elif train_mode in ['rank','auto']:
            self.choice_emb = torch.zeros(self.vocab_size, len(self.dim_list)).to(self.device)
            if train_mode in ['auto']:
                # high, low twins weight
                self.rank_e_weight = torch.tensor([self.generate_probability_distribution(x) for x in self.count_e],device=self.device)
                if self.configuration_key.split('.')[-1] == 'entity_embedder':
                    self.picker = Picker(self.config, dataset, self.dim, max(self.rank)+1,self.space)
                    if self.adae_config['cie']:
                        self.dim_list = [i for i in range(1,int((self.dim)/2+1))]
                    self.all_indexes = torch.arange(
                        self.vocab_size, dtype=torch.long, device=self.device
                        )
                    self.dim_l = torch.tensor(self.dim_list,device=self.device).expand(self.all_indexes.shape[0], -1)
                # self.picker_low = Picker(self.config, dataset, self.dim, max(self.rank)+1,self.space)
                
                
                if self.adae_config['cie']:
                    self.choice_emb = torch.zeros(self.vocab_size, self.dim).to(self.device)
                    

                    if self.space=='complex':
                        half_size = int(self.dim/2)
                        self.choice_emb = torch.zeros(self.vocab_size, half_size).to(self.device)
                        self.mask =  torch.cat( (torch.tril(torch.ones(half_size,half_size,device=self.device)),torch.tril(torch.ones(half_size,half_size,device=self.device)) ),dim = -1)
                    else:
                        self.mask =  torch.tril(torch.ones(self.dim,self.dim,device=self.device))
                else:
                    self.mask =  torch.zeros(len(self.dim_list),self.dim,device=self.device)
                    for i, k in enumerate(self.dim_list):
                        hk = int(k/2)
                        half_size = int(self.dim/2)
                        self.mask[i, :hk] = 1
                        self.mask[i, half_size:half_size+hk] = 1
            self.choice_emb[:,-1] = 1         
            
            self.t_s = self.adae_config['t_s']*2
            

            
            
            # aligment way init
            if self.adae_config['ali_way'] == 'ts':
                self.Transform_layer_list =nn.ModuleList([nn.Linear(self.dim_list[i],self.dim) for i in range(0,len(self.dim_list))])
                # self.Transform_layer_list_1 =nn.ModuleList([nn.Linear(self.dim,self.dim) for i in range(0,len(self.dim_list))])
                for i in range(0,len(self.dim_list)):
                    nn.init.xavier_uniform_(self.Transform_layer_list[i].weight.data)
                    # nn.init.xavier_uniform_(self.Transform_layer_list_1[i].weight.data)
            elif self.adae_config['ali_way'] == 'zp':
                assert self.dim >= max(self.dim_list)

            # embedding-list shard init
            if self.adae_config['share']:
                # share 不需要emb-list
                self._embeddings_clone = torch.nn.Embedding(self.vocab_size, self.dim, sparse=self.sparse,)
                self._embeddings_clone.weight.requires_grad = False
                pass
            else:  
                if self.configuration_key.split('.')[-1] == 'relation_embedder':
                    # self._embeddings.weight.requires_grad = False
                    pass
                else:
                    self._embeddings_list =nn.ParameterList( [nn.Parameter(torch.zeros(self.vocab_size, i)) for i in self.dim_list])

                    if not init_for_load_only:
                        for i in range(len(self.dim_list)):
                            self.initialize(self._embeddings_list[i].data)
                        self._normalize_embeddings_list()
              
        else:
            raise ValueError(f"Invalid value train_mode={self.adae_config['train_mode']}")
       
    def _adaE(self, **kwargs) -> KgeEmbedder:
        kwargs['all'] = kwargs.get('all', False)
        indexes = kwargs.get('indexes', None)
        if_training = kwargs.get('training', True)
        step = kwargs.get("step",10000)
        #判断是否是relation_embedder，是就直接用size最大值
        if self.configuration_key.split('.')[-1] == 'relation_embedder':
            emb = self._embeddings(indexes)
        elif self.adae_config['train_mode'] in ['original', 'fix', 'rank','auto','retrain']:
            if self.adae_config['train_mode'] == 'original':
                # original means no change
                emb = self._embeddings(indexes)
            elif self.adae_config['train_mode'] == 'fix':
                # fix means all dim is same
                emb = self._embeddings(indexes)
                # emb = self.Transform_layer(emb)
                emb = self.BN(self.Transform_layer1((self.BN(torch.relu((self.Transform_layer(emb)))))))
            elif self.adae_config['train_mode'] == 'rank':
                pro = -1
                if self.adae_config['folder']!='None':
                    #代表在重训练
                    pro = self._picker_retrain(indexes)
                else:
                    pro = self._picker_rank(indexes)
                emb = self._aligment_fix(indexes, probability=pro, ali_way=self.adae_config['ali_way'])
             
            elif self.adae_config['train_mode'] == 'auto':
                # rank means use learned choice of dim with each entity 
                if if_training:
                    if self.adae_config['no_picker']:
                        # frequency caterage
                        pro = self.picker.bucket(self.rank[indexes])
                    else:
                        pro = self._picker(indexes)
                    # a random choice trigger with pro of 0.1
                    # rate = max(0.1,1*np.exp(-0.0003*self.step))
                    # if random.random()<0.02:
                    #     t = torch.softmax(torch.rand(len(indexes),len(self.adae_config['dim_list']),device=self.device),dim=1)
                    #     _, max_indices = torch.max(t, dim=1)
                    #     pro = F.one_hot(max_indices, num_classes=len(self.adae_config['dim_list'])).float()
                else:
                    pro = self._picker_fix(indexes)
                
                if self.adae_config['cie']:
                    emb = self._aligment_cie(indexes, if_training, probability=pro, ali_way=self.adae_config['ali_way'],step = step)
                else:
                    emb = self._aligment(indexes, if_training, probability=pro, ali_way=self.adae_config['ali_way'],step = step)


        return emb
   
    def _picker_rank(self, indexes):
        """
        using pre frequency as choice of entity embedding size despite r
        """          
        batch_size=len(indexes)
        # 上次的emb，离散选择
        # label = torch.tensor([0 for i in range(batch_size)],device=self.device).unsqueeze(-1)
        label  =  self.rank[indexes].unsqueeze(-1)
        pro = torch.zeros((batch_size,len(self.dim_list)),device = self.device).scatter_(1, label, 1)    
                
        return pro
    def _picker_retrain(self, indexes):

        pro = self.choice[indexes]
        return pro
    
    def _picker(self, indexes):
        """
        picker, when training
        """          
        # print(max(self.rank))
        # print(self.picker.bucket.weight.shape)
        if self.adae_config['share']:
            emb = self._embeddings_clone(indexes).detach()
        else:
            emb = self._embeddings(indexes).detach()
        input_h = torch.cat((self.picker.bucket(self.rank[indexes]),emb),dim=1)
        # input_h = self.picker.bucket(self.rank[indexes])
        pro = self.picker(input_h,self.rank_e_weight[indexes])
        # pro =  o*self.rank_e_weight[indexes][:,1].unsqueeze(1)# high
        # o_low = self.picker_low(input_h)
        # pro_low = o_low *self.rank_e_weight[indexes][:,0].unsqueeze(1)# low
        
        final_pro = pro
        
        
        return final_pro
    def _picker_fix(self, indexes):
        """
        picker_fix, when eval
        """          
        pro =  self.choice_emb[indexes]
        return pro

    def _aligment_fix(self, indexes, probability=None, ali_way='ts'):
        """
        fixed ,which means no gumbel softmax/ used in rank mode
        """
        emb = []
        for i in range(0,len(self.dim_list)):     
            if self.adae_config['share']:
                emb.append(self._embeddings(indexes)[:,:int(self.dim_list[i])])
            else:
                emb.append(self._embeddings_list[i][indexes])  # [bs, 1, dim]
        output = []
        for i in range(0,len(self.dim_list)):
            output_pre = -1
            if ali_way == 'ts':       
                if self.dim_list[i]==self.dim:
                # if i==0:
                    output_pre = emb[i]
                else:
                    output_pre = (self.Selection(self.Transform_layer_list[i](emb[i])))
            elif ali_way=='zp':
                 output_pre = self._zero_padding(emb[i])
            output.append(output_pre)
        # 堆叠以便于计算
        emb = torch.stack(output, dim=1)
        # no Gumbal softmax probability
        Gpro = probability
        #soft selection   
        emb_final = torch.mul(emb, Gpro.unsqueeze(-1)).sum(dim = 1)
        with torch.no_grad():
            # save fianl emb
            if not self.is_bilevel:
                # self._embeddings.weight.data[indexes] = emb_final
                self.choice_emb[indexes] = probability

        return emb_final
  
    
    def _aligment(self, indexes,  if_training, probability=None,  ali_way='ts',step = 10000):
        """
        align with gumbal softmax / only be used in auto modef
        """
  
        Tau=max(0.2,1*np.exp(-0.0001*step))
        
        emb = []
        for i in range(0,len(self.dim_list)): 
            if self.adae_config['share']:
                emb.append(self._embeddings(indexes)[:,:int(self.dim_list[i])])
            else:    
                emb.append(self._embeddings_list[i][indexes])  # [bs, 1, dim]
        output = []
        for i in range(0,len(self.dim_list)): 
            output_pre = -1   
            if  ali_way == 'ts':
                output_pre = (self.Selection(self.Transform_layer_list[i](emb[i])))
            elif ali_way=='zp':
                 output_pre = self._zero_padding(emb[i])
            output.append(output_pre)
        # 堆叠以便于计算
        head_s = torch.stack(output, dim=1)
        # Gumbal softmax probability
        if if_training:
            Gpro = F.gumbel_softmax(probability, tau=Tau, hard=False)
        else:
            Gpro =  torch.zeros(probability.shape,device=self.device)
            Gpro_index = torch.argmax(probability, dim = -1).unsqueeze(-1)
            Gpro = Gpro.scatter_(1, Gpro_index, 1) 
            
        #soft selection   
        emb_final = torch.mul(head_s, Gpro.unsqueeze(-1)).sum(dim = 1)
        with torch.no_grad():
            # save fianl emb
            if not self.is_bilevel:
                self._embeddings.weight.data[indexes] = emb_final
                self.choice_emb[indexes] = probability

        return emb_final
    # def _aligment(self, indexes,  if_training, probability=None,  ali_way='ts',step = 10000):
    #     """
    #     align with gumbal softmax / only for continuous input embeddings size, no ts
    #     """
    #     Tau=max(0.5,1*np.exp(-0.00003*step))
    #     if if_training:
    #         Gpro = F.gumbel_softmax(probability, tau=Tau, hard=True)
    #         # Gpro = torch.softmax(probability, dim = -1)
    #     else:
    #         Gpro =  torch.zeros(probability.shape,device=self.device)
    #         Gpro_index = torch.argmax(probability, dim = -1).unsqueeze(-1)
    #         Gpro = Gpro.scatter_(1, Gpro_index, 1) 
    #     mask = torch.matmul(Gpro,self.mask)
    #     a = self.adae_config['padding']
    #     paddings = (torch.ones(mask.shape[0],mask.shape[1],device=self.device)-mask)*a
    #     emb = self._embeddings(indexes)
    #     emb_final = emb*mask +paddings
    #     with torch.no_grad():
    #         # save fianl emb
    #         if not self.is_bilevel:
    #             self._embeddings.weight.data[indexes] = emb_final
    #             self.choice_emb[indexes] = probability
    #     return emb_final
    def _aligment_cie(self, indexes,  if_training, probability=None,  ali_way='ts',step = 10000):
        """
        align with gumbal softmax / only for continuous input embeddings size
        """
        Tau=max(0.5,1*np.exp(-0.0003*step))
        if if_training:
            Gpro = F.gumbel_softmax(probability, tau=Tau, hard=True)
            # Gpro = self.topk(probability, tau=Tau, k=1)
        else:
            Gpro =  torch.zeros(probability.shape,device=self.device)
            Gpro_index = torch.argmax(probability, dim = -1).unsqueeze(-1)
            Gpro = Gpro.scatter_(1, Gpro_index, 1) 
        mask = torch.matmul(Gpro,self.mask)
        
        paddings = (torch.ones(mask.shape[0],mask.shape[1],device=self.device)-mask)*self.adae_config['padding']
        emb = self._embeddings(indexes)
        #soft selection   
        emb_final = emb*mask +paddings
        # emb_final = emb
        with torch.no_grad():
            pass
            # save fianl emb
            # if not self.is_bilevel:
            #     self._embeddings_clone.weight.data[indexes] = emb_final
            #     self.choice_emb[indexes] = probability

        return emb_final

    def _aligment_zp(self, indexes,  if_training, probability=None,  ali_way='ts',step = 10000):
        """
        align with gumbal softmax / only for continuous input embeddings size in dim_list 
        """
        Tau=max(0.5,1*np.exp(-0.0003*step))

        # Tau = 0.1
        if if_training:
            Gpro = F.gumbel_softmax(probability, tau=Tau, hard=True)
            # Gpro = torch.softmax(probability, dim = -1)
        else:
            Gpro =  torch.zeros(probability.shape,device=self.device)
            Gpro_index = torch.argmax(probability, dim = -1).unsqueeze(-1)
            Gpro = Gpro.scatter_(1, Gpro_index, 1) 
        emb = self._embeddings(indexes)
      

        
        mask = torch.matmul(Gpro,self.mask)
        a = self.adae_config['padding']
        paddings = (torch.ones(mask.shape[0],mask.shape[1],device=self.device)-mask)*a
        # emb_final = emb
        emb_final = emb*mask +paddings
        with torch.no_grad():
            if not self.is_bilevel:
                # no need
                if self.adae_config['share']:
                    self._embeddings_clone.weight.data[indexes] = emb_final
                else:
                    self._embeddings.weight.data[indexes] = emb_final
                self.choice_emb[indexes] = probability

        return emb_final
    
    def _zero_padding(self, emb):        
        padding_size = abs(emb.shape[1]-self.dim)
        a = self.config.get("multi_lookup_embedder.initialize_args.uniform_.a")
        if self.space=='complex':
            if padding_size==0:  
                return emb
            else:
                half_emb_size = int(emb.shape[1]/2)
                half_padding_size = int(padding_size/2)
                x = torch.ones((emb.shape[0], half_padding_size),device=self.device)
                
                x = x*(0)
                emb = torch.cat((emb[:,:half_emb_size],x,emb[:,half_emb_size:],x),dim=1)
                
                return emb
            

        else:
            emb = torch.cat((emb,torch.ones((emb.shape[0], padding_size),device = self.device)*a),dim=1)

        return emb
    

    def generate_probability_distribution(self, frequency):
        # 定义转折点
        turning_point = self.pivot

        if frequency <= turning_point :
            # 低频部分，可以根据需要自定义低频部分的概率分布
            probability = frequency / (2 * turning_point)
            # probability = 0
        else:
            # 高频部分，可以根据需要自定义高频部分的概率分布
            probability = min(1, (frequency - turning_point) / (turning_point * 2))
            # probability= 1
        # 高、低
        # return [1-probability,probability]
        return [0,1]
    
    def topk(self, logits: Tensor, tau: float = 1, hard: bool = True, eps: float = 1e-10, dim: int = -1, k = 1) -> Tensor:

        gumbels = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
        )  # ~Gumbel(0,1)
        gumbels = (logits ) / tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(dim)

        if hard:
            # Straight through.
            topk_values, topk_indices = torch.topk(y_soft, k, dim=dim)
            selected_index = torch.randint(k, (topk_indices.shape[0],))
            index = topk_indices[:, selected_index]
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
            ret = (y_hard - y_soft).detach() + y_soft
        else:
            # Reparametrization trick.
            ret = y_soft
        return ret







class Picker(nn.Module):
    def __init__( self, config: Config, dataset: Dataset, dim, bucket_size,space = 'complex') -> None:
        # # father init
        super(Picker, self).__init__()
        self.adae_config = config.options['AdaE_config']
        self.device = config.get("job.device")
        self.dim: int = dim
        self.space = space
        self.dim_list_size = len(self.adae_config['dim_list'])

        if self.adae_config['no_picker']:
            self.dim_bucket = self.dim_list_size
        else:
            self.dim_bucket = int(128)
        if self.adae_config['cie']:
            self.dim_bucket = 2
            #目前的输入之后fre
            self.FC1 = nn.Linear(self.dim_bucket+self.dim,128).to(self.device)
            if self.space=='complex':
                h=int(self.dim/2)
            else:
                h=self.dim
            self.FC2 = nn.Linear(128,h).to(self.device)
            self.FC1_1 = nn.Linear(self.dim_bucket+self.dim,128).to(self.device)
            self.FC2_1 = nn.Linear(128,h).to(self.device)
        else:  
            self.FC1 = nn.Linear(self.dim_bucket,20).to(self.device)
            self.FC2 = nn.Linear(20,self.dim_list_size).to(self.device)
            self.FC1_1 = nn.Linear(self.dim_bucket,20).to(self.device)
            self.FC2_1 = nn.Linear(20,self.dim_list_size).to(self.device)
            # self.FC3 = nn.Linear(64,self.dim_list_size).to(self.device)
        nn.init.xavier_uniform_(self.FC1.weight.data)
        nn.init.xavier_uniform_(self.FC2.weight.data)
        nn.init.xavier_uniform_(self.FC1_1.weight.data)
        nn.init.xavier_uniform_(self.FC2_1.weight.data)
        # nn.init.xavier_uniform_(self.FC3.weight.data)  
        self.Picker = nn.Sequential(
            self.FC1,
            nn.Dropout(0.2),
            nn.ReLU(),
            self.FC2,
            # nn.Dropout(0.2),
            # nn.ReLU(),
            # self.FC3
        ) 
        self.Picker_1 = nn.Sequential(
            self.FC1_1,
            nn.Dropout(0.2),
            nn.ReLU(),
            self.FC2_1,
            # nn.Dropout(0.2),
            # nn.ReLU(),
            # self.FC3
        ) 
        
        # bucket emb
        self.k = bucket_size
        self.bucket = nn.Embedding(int(self.k), self.dim_bucket).to(self.device)
        # self.bucket.weight.requires_grad = False
        if self.adae_config['no_picker']:
            self.bucket.weight.data= torch.zeros(self.k,self.dim_bucket,device=self.device)
            self.bucket.weight.data[:,-1]=1
        else:
            nn.init.uniform_(tensor=self.bucket.weight.data,a=-0.281)
        
    #
    def forward(self, input, weight):
        # 上次的emb，离散选择
        pro = self.Picker(input)*weight[:,1].unsqueeze(1)
        # pro_1 = self.Picker_1(input)*weight[:,0].unsqueeze(1)
        return pro
    
    