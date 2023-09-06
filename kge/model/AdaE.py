import torch
from kge import Config, Dataset
from kge.model.kge_model import KgeEmbedder, RelationalScorer, KgeModel


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
    def get_s_embedder(self) -> KgeEmbedder:
            return self._multi_entity_embedder
    def get_o_embedder(self) -> KgeEmbedder:
            return self._multi_entity_embedder
    def get_p_embedder(self) -> KgeEmbedder:
            return self._multi_relation_embedder
    
class Picker(AdaE):
    def __init__( self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,) -> None:
        
        if args.train_mode in ['auto']:
            self.dim_bucket = int(self.Transfromed_dim/8)
            self.FC1 = nn.Linear(self.dim_bucket+self.entity_dim,128).to(DEVICE)
            self.FC2 = nn.Linear(128,64).to(DEVICE)
            self.FC3 = nn.Linear(64,self.dim_list_size).to(DEVICE)
            nn.init.xavier_uniform_(self.FC1.weight.data)
            nn.init.xavier_uniform_(self.FC2.weight.data)
            nn.init.xavier_uniform_(self.FC3.weight.data)  
            self.Controller = nn.Sequential(
                self.FC1,
                nn.Dropout(0.5),
                nn.Tanh(),
                self.FC2,
                nn.Dropout(0.5),
                nn.Tanh(),
                self.FC3
            ) 
            self.FC1_r = nn.Linear(self.dim_bucket+self.relation_dim,128).to(DEVICE)
            self.FC2_r = nn.Linear(128,64).to(DEVICE)
            self.FC3_r = nn.Linear(64,self.dim_list_size).to(DEVICE)
            nn.init.xavier_uniform_(self.FC1_r.weight.data)
            nn.init.xavier_uniform_(self.FC2_r.weight.data)
            nn.init.xavier_uniform_(self.FC3_r.weight.data)
            self.Controller_r = nn.Sequential(
                self.FC1_r,
                nn.Dropout(0.5),
                nn.Tanh(),
                self.FC2_r,
                nn.Dropout(0.5),
                nn.Tanh(),
                self.FC3_r
                )

            # bucket emb
            self.k = len(self.emb_dim_list)
            self.bucket = nn.Embedding(self.k, self.dim_bucket)
         

