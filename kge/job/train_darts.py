import time
import torch
import torch.utils.data
from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.job.train_negative_sampling import TrainingJobNegativeSampling
from kge.job.train_1vsAll import TrainingJob1vsAll
from kge.job.train_KvsAll import TrainingJobKvsAll
from kge.util import KgeSampler
from kge.model.transe import TransEScorer
import itertools
import os
import math
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
import torch
import torch.utils.data

from kge.job import Job, TrainingOrEvaluationJob
from kge.model import KgeModel, Architect
from kge.util import KgeLoss, KgeOptimizer, KgeSampler, KgeLRScheduler
from kge.util.io import load_checkpoint
from kge.job.trace import format_trace_entry
from typing import Any, Callable, Dict, List, Optional
import kge.job.util
from kge.util.metric import Metric
from kge.misc import init_from
SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]



class TrainingJobDarts(TrainingJobNegativeSampling, TrainingJobKvsAll, TrainingJob1vsAll ):
    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        # need optimizer_c
        self.adae_config = self.config.options['AdaE_config']
        self.lr_trans = self.adae_config["lr_trans"]
        if not self.is_forward_only:
            if self.adae_config['train_mode'] in ['original']:
                pass
            elif self.adae_config['train_mode'] in ['fix']:
                embeddings_params_e = self.model._base_model._entity_embedder._embeddings.parameters()
                embeddings_params_r = self.model._base_model._relation_embedder._embeddings.parameters()
                embeddings_params_e_id = list( map( id, embeddings_params_e ))
                embeddings_params_r_id = list( map( id, embeddings_params_r ))
                trans_e = self.model._base_model._entity_embedder.Transform_layer.parameters()
                trans_r = self.model._base_model._relation_embedder.Transform_layer.parameters()
                trans_e_id = list( map( id, trans_e ) )
                trans_r_id = list( map( id, trans_r ) )
                BN_e_id = list(map( id, self.model._base_model._entity_embedder.BN.parameters() )) 
                BN_r_id = list(map( id, self.model._base_model._relation_embedder.BN.parameters() ))
                # params
                base_params = filter(lambda p: id(p) not in trans_e_id+trans_r_id+BN_e_id+BN_r_id, self.model.parameters())
                trans_params = filter(lambda p: id(p) not in embeddings_params_e_id+embeddings_params_r_id, self.model.parameters())
                opt = getattr(torch.optim, config.get("train.optimizer.default.type"))
                self.optimizer = opt(
                    [{'params':base_params},
                     {'params':trans_params, 'lr': self.lr_trans},
                     ], **config.get("train.optimizer.default.args") 
                    ) #用来更新theta的optimizer
                
                self.kge_lr_scheduler = KgeLRScheduler(config, self.optimizer)
                self._lr_warmup = self.config.get("train.lr_warmup")
                for group in self.optimizer.param_groups:
                    group["initial_lr"]=group["lr"]
            elif self.adae_config['train_mode'] in ['rank']:
                pass
            elif self.adae_config['train_mode'] in ['auto']:
                picker_e = self.model._entity_embedder.picker
                picker_r = self.model._relation_embedder.picker
                params_p = list(picker_e.bucket.parameters()) + list(picker_e.FC1.parameters())+list(picker_e.FC2.parameters())+list(picker_e.FC3.parameters())+list(picker_r.bucket.parameters()) + list(picker_r.FC1.parameters())+list(picker_r.FC2.parameters())+list(picker_r.FC3.parameters())
                params_p_id = list(map( id,picker_e.bucket.parameters())) + list(map( id,picker_e.FC1.parameters()))+list(map( id,picker_e.FC2.parameters()))+list(map( id,picker_e.FC3.parameters()))+list(map( id,picker_r.bucket.parameters())) + list(map( id,picker_r.FC1.parameters()))+list(map( id,picker_r.FC2.parameters()))+list(map( id,picker_r.FC3.parameters()))

                # picker =  {'e':picker_e,'r':picker_r}
                # learnable_parameters = [param for name, param in vars(picker).items() if isinstance(param, torch.nn.Parameter) and param.requires_grad]
                embeddings_params_e = list( map( id, self.model._base_model._entity_embedder._embeddings.parameters() ) )
                embeddings_params_r = list( map( id, self.model._base_model._relation_embedder._embeddings.parameters() ) )

                base_params = filter(lambda p: id(p) not in embeddings_params_r+embeddings_params_e+params_p_id, self.model.parameters())


                opt = getattr(torch.optim, config.get("train.optimizer.default.type"))
                self.optimizer = opt(
                    [{'params':base_params}], **config.get("train.optimizer.default.args") ) #用来更新theta的optimizer
                self.kge_lr_scheduler = KgeLRScheduler(config, self.optimizer)
                self._lr_warmup = self.config.get("train.lr_warmup")
                for group in self.optimizer.param_groups:
                    group["initial_lr"]=group["lr"]

                
                self.optimizer_p = opt(
                    [{'params':params_p,'name':'default'}],  **config.get("train.optimizer.default.args")
                        ) #用来更新theta的optimizer
                self.kge_lr_scheduler_p = KgeLRScheduler(config, self.optimizer_p)
                self._lr_warmup = self.config.get("train.lr_warmup")
                for group in self.optimizer_p.param_groups:
                    group["initial_lr"]=group["lr"]

                self.architect = Architect(self.model, params_p, self.optimizer_p, self, self.adae_config)
    
    def _get_collate_fun(self, mode='ng_sample'):

        mode_list = ['ng_sample', '1vsall', 'kvsall']
        if mode == mode_list[0]:
            # create the collate function
            def collate(batch):
                """For a batch of size n, returns a tuple of:

                - triples (tensor of shape [n,3], ),
                - negative_samples (list of tensors of shape [n,num_samples]; 3 elements
                in order S,P,O)
                """
                triples = self.dataset.split(self.train_split)[batch, :].long()
                # labels = torch.zeros((len(batch), self._sampler.num_negatives_total + 1))
                # labels[:, 0] = 1
                # labels = labels.view(-1)

                negative_samples = list()
                for slot in [S, P, O]:
                    negative_samples.append(self._sampler.sample(triples, slot))
                triples_t, triples_v = None, None
                negative_samples_t, negative_samples_v =  list(),  list() 
                
                ratio = self.adae_config['ratio']
                s_u = self.adae_config['s_u']
                if s_u:
                    # split data for darts
                    triples_t = triples[:int(ratio*triples.shape[0])]
                    triples_v = triples[int(ratio*triples.shape[0]):]
                    for slot in [S, P, O]:
                        negative_samples_t.append(self._sampler.sample(triples_t, slot))
                        negative_samples_v.append(self._sampler.sample(triples_v, slot))

                    # return {"triples": triples, "negative_samples": negative_samples,  "triples_t": triples_t, "triples_v": triples_v, "negative_samples_t": negative_samples_t, "negative_samples_v": negative_samples_v}
                return [{"triples": triples_t, "negative_samples": negative_samples_t},  {"triples": triples_v, "negative_samples": negative_samples_v}]
            return collate 
        elif mode== mode_list[1]:
             # create the collate function
            def collate(batch):
                """For a batch of size n, returns a dictionary of:

                - queries: nx2 tensor, row = query (sp, po, or so indexes)
                - label_coords: for each query, position of true answers (an Nx2 tensor,
                first columns holds query index, second colum holds index of label)
                - query_type_indexes (vector of size n holding the query type of each query)
                - triples (all true triples in the batch; e.g., needed for weighted
                penalties)

                """

                # count how many labels we have across the entire batch
                num_ones = 0
                for example_index in batch:
                    start = 0
                    for query_type_index in range(len(self.query_types)):
                        end = self.query_last_example[query_type_index]
                        if example_index < end:
                            example_index -= start
                            num_ones += self.query_indexes[query_type_index]._values_offset[
                                example_index + 1
                            ]
                            num_ones -= self.query_indexes[query_type_index]._values_offset[
                                example_index
                            ]
                            break
                        start = end

                # now create the batch elements
                queries_batch = torch.zeros([len(batch), 2], dtype=torch.long)
                query_type_indexes_batch = torch.zeros([len(batch)], dtype=torch.long)
                label_coords_batch = torch.zeros([num_ones, 2], dtype=torch.int)
                triples_batch = torch.zeros([num_ones, 3], dtype=torch.long)
                current_index = 0
                for batch_index, example_index in enumerate(batch):
                    start = 0
                    for query_type_index, query_type in enumerate(self.query_types):
                        end = self.query_last_example[query_type_index]
                        if example_index < end:
                            example_index -= start
                            query_type_indexes_batch[batch_index] = query_type_index
                            queries = self.query_indexes[query_type_index]._keys
                            label_offsets = self.query_indexes[
                                query_type_index
                            ]._values_offset
                            labels = self.query_indexes[query_type_index]._values
                            if query_type == "sp_":
                                query_col_1, query_col_2, target_col = S, P, O
                            elif query_type == "s_o":
                                query_col_1, target_col, query_col_2 = S, P, O
                            else:
                                target_col, query_col_1, query_col_2 = S, P, O
                            break
                        start = end

                    queries_batch[batch_index,] = queries[example_index]
                    start = label_offsets[example_index]
                    end = label_offsets[example_index + 1]
                    size = end - start
                    label_coords_batch[
                        current_index : (current_index + size), 0
                    ] = batch_index
                    label_coords_batch[current_index : (current_index + size), 1] = labels[
                        start:end
                    ]
                    triples_batch[
                        current_index : (current_index + size), query_col_1
                    ] = queries[example_index][0]
                    triples_batch[
                        current_index : (current_index + size), query_col_2
                    ] = queries[example_index][1]
                    triples_batch[
                        current_index : (current_index + size), target_col
                    ] = labels[start:end]
                    current_index += size

                # all done
                return {
                    "queries": queries_batch,
                    "label_coords": label_coords_batch,
                    "query_type_indexes": query_type_indexes_batch,
                    "triples": triples_batch,
                }

            return collate
        elif mode== mode_list[2]:
            return 


    def run_epoch(self) -> Dict[str, Any]:
        """ Runs an epoch and returns its trace entry. """
        
        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type=self.type_str,
            scope="epoch",
            epoch=self.epoch,
            split=self.train_split,
            batches=len(self.loader),
            size=self.num_examples,
        )
        if not self.is_forward_only:
            self.current_trace["epoch"].update(
                lr=[group["lr"] for group in self.optimizer.param_groups],
            )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        # variables that record various statitics
        sum_loss = 0.0
        sum_penalty = 0.0
        sum_penalties = defaultdict(lambda: 0.0)
        epoch_time = -time.time()
        prepare_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        optimizer_time = 0.0

        # process each batch
        for batch_index, batch in enumerate(self.loader):
            batch_t, batch_v = batch[0], batch[1]
            # create initial batch trace (yet incomplete)
            self.current_trace["batch"] = {
                "type": self.type_str,
                "scope": "batch",
                "epoch": self.epoch,
                "split": self.train_split,
                "batch": batch_index,
                "batches": len(self.loader),
            }
            if not self.is_forward_only:
                self.current_trace["batch"].update(
                    lr=[group["lr"] for group in self.optimizer.param_groups],
                )

            # run the pre-batch hooks (may update the trace)
            for f in self.pre_batch_hooks:
                f(self)

            
            

            # process batch (preprocessing + forward pass + backward pass on loss)
            done = False
            while not done:
                    try:
                        # try running the batch
                        # architecture的更新
                        if self.adae_config['train_mode'] in ['auto']:   
                            
                            #对α进行更新，对应伪代码的第一步，也就是用公式6
                            if batch_index % self.adae_config['s_u'] == 0:
                                self.architect.step(batch_index, batch_t, batch_v, self.adae_config['lr_p'], self.optimizer)
                            
                        if not self.is_forward_only:
                            self.optimizer.zero_grad()
                        batch_result: TrainingJob._ProcessBatchResult = self._process_batch(
                            batch_index, batch_t
                        )
                        done = True
                    except RuntimeError as e:
                        # is it a CUDA OOM exception and are we allowed to reduce the
                        # subbatch size on such an error? if not, raise the exception again
                        if (
                            "CUDA out of memory" not in str(e)
                            or not self._subbatch_auto_tune
                        ):
                            raise e

                        # try rerunning with smaller subbatch size
                        tb = traceback.format_exc()
                        self.config.log(tb)
                        self.config.log(
                            "Caught OOM exception when running a batch; "
                            "trying to reduce the subbatch size..."
                        )

                        if self._max_subbatch_size <= 0:
                            self._max_subbatch_size = self.batch_size
                        if self._max_subbatch_size <= 1:
                            self.config.log(
                                "Cannot reduce subbatch size "
                                f"(current value: {self._max_subbatch_size})"
                            )
                            raise e  # cannot reduce further

                        self._max_subbatch_size //= 2
                        self.config.set(
                            "train.subbatch_size", self._max_subbatch_size, log=True
                        )
            sum_loss += batch_result.avg_loss * batch_result.size
            # self.optimizer.step()
            # self.optimizer.zero_grad()
            # determine penalty terms (forward pass)
            batch_forward_time = batch_result.forward_time - time.time()
            penalties_torch = self.model.penalty(
                epoch=self.epoch,
                batch_index=batch_index,
                num_batches=len(self.loader),
                batch=batch_t,
            )
            batch_forward_time += time.time()

            # backward pass on penalties
            batch_backward_time = batch_result.backward_time - time.time()
            penalty = 0.0
            for index, (penalty_key, penalty_value_torch) in enumerate(penalties_torch):
                if not self.is_forward_only:
                    penalty_value_torch.backward()
                penalty += penalty_value_torch.item()
                sum_penalties[penalty_key] += penalty_value_torch.item()
            sum_penalty += penalty
            batch_backward_time += time.time()

            # determine full cost
            cost_value = batch_result.avg_loss + penalty

            # abort on nan
            if self.abort_on_nan and math.isnan(cost_value):
                raise FloatingPointError("Cost became nan, aborting training job")

            # TODO # visualize graph
            # if (
            #     self.epoch == 1
            #     and batch_index == 0
            #     and self.config.get("train.visualize_graph")
            # ):
            #     from torchviz import make_dot

            #     f = os.path.join(self.config.folder, "cost_value")
            #     graph = make_dot(cost_value, params=dict(self.model.named_parameters()))
            #     graph.save(f"{f}.gv")
            #     graph.render(f)  # needs graphviz installed
            #     self.config.log("Exported compute graph to " + f + ".{gv,pdf}")

            # print memory stats
            if self.epoch == 1 and batch_index == 0:
                if self.device.startswith("cuda"):
                    self.config.log(
                        "CUDA memory after first batch: allocated={:14,} "
                        "reserved={:14,} max_allocated={:14,}".format(
                            torch.cuda.memory_allocated(self.device),
                            torch.cuda.memory_reserved(self.device),
                            torch.cuda.max_memory_allocated(self.device),
                        )
                    )

            # update parameters
            batch_optimizer_time = -time.time()
            if not self.is_forward_only:
                self.optimizer.step()
            batch_optimizer_time += time.time()

            # update batch trace with the results
            self.current_trace["batch"].update(
                {
                    "size": batch_result.size,
                    "avg_loss": batch_result.avg_loss,
                    "penalties": [p.item() for k, p in penalties_torch],
                    "penalty": penalty,
                    "cost": cost_value,
                    "prepare_time": batch_result.prepare_time,
                    "forward_time": batch_forward_time,
                    "backward_time": batch_backward_time,
                    "optimizer_time": batch_optimizer_time,
                    "event": "batch_completed",
                }
            )

            # run the post-batch hooks (may modify the trace)
            for f in self.post_batch_hooks:
                f(self)

            # output, then clear trace
            if self.trace_batch:
                self.trace(**self.current_trace["batch"])
            self.current_trace["batch"] = None

            # print console feedback
            self.config.print(
                (
                    "\r"  # go back
                    + "{}  batch{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}"
                    + ", avg_loss {:.4E}, penalty {:.4E}, cost {:.4E}, time {:6.2f}s"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_index,
                    len(self.loader) - 1,
                    batch_result.avg_loss,
                    penalty,
                    cost_value,
                    batch_result.prepare_time
                    + batch_forward_time
                    + batch_backward_time
                    + batch_optimizer_time,
                ),
                end="",
                flush=True,
            )

            # update epoch times
            prepare_time += batch_result.prepare_time
            forward_time += batch_forward_time
            backward_time += batch_backward_time
            optimizer_time += batch_optimizer_time

        # all done; now trace and log
        epoch_time += time.time()
        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back

        other_time = (
            epoch_time - prepare_time - forward_time - backward_time - optimizer_time
        )

        # add results to trace entry
        self.current_trace["epoch"].update(
            dict(
                avg_loss=sum_loss / self.num_examples,
                avg_penalty=sum_penalty / len(self.loader),
                avg_penalties={
                    k: p / len(self.loader) for k, p in sum_penalties.items()
                },
                avg_cost=sum_loss / self.num_examples + sum_penalty / len(self.loader),
                epoch_time=epoch_time,
                prepare_time=prepare_time,
                forward_time=forward_time,
                backward_time=backward_time,
                optimizer_time=optimizer_time,
                other_time=other_time,
                event="epoch_completed",
            )
        )

        # run hooks (may modify trace)
        for f in self.post_epoch_hooks:
            f(self)

        # output the trace, then clear it
        trace_entry = self.trace(**self.current_trace["epoch"], echo=False, log=True)
        self.config.log(
            format_trace_entry("train_epoch", trace_entry, self.config), prefix="  "
        )
        self.current_trace["epoch"] = None

        return trace_entry
    def save_to(self, checkpoint: Dict) -> Dict:
        """used in darts"""
        train_mode = self.adae_config['train_mode']
        if train_mode in ['original','fix','rank']:
            train_checkpoint = {
                "type": "train",
                "epoch": self.epoch,
                "valid_trace": self.valid_trace,
                "model": self.model.save(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.kge_lr_scheduler.state_dict(),
                "job_id": self.job_id,
            }
        elif train_mode == 'auto':
            train_checkpoint = {
                "type": "train",
                "epoch": self.epoch,
                "valid_trace": self.valid_trace,
                "model": self.model.save(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "optimizer_p_state_dict": self.optimizer_p.state_dict(),
                "lr_scheduler_state_dict": self.kge_lr_scheduler.state_dict(),
                "lr_scheduler_p_state_dict": self.kge_lr_scheduler_p.state_dict(),
                "job_id": self.job_id,
            }
        train_checkpoint = self.config.save_to(train_checkpoint)
        checkpoint.update(train_checkpoint)
        return checkpoint
