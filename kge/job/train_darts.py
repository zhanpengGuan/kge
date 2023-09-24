import time
import torch
import torch.utils.data
from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.job.train_negative_sampling import TrainingJobNegativeSampling
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



class TrainingJobDarts(TrainingJobNegativeSampling):
    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        # need optimizer_c
        self.adae_config = self.config.options['AdaE_config']
        
        
        if not self.is_forward_only:
            picker_e = self.model._entity_embedder.picker
            picker_r = self.model._relation_embedder.picker
            params_p = list(picker_e.bucket.parameters()) + list(picker_e.FC1.parameters())+list(picker_e.FC2.parameters())+list(picker_e.FC3.parameters())+list(picker_r.bucket.parameters()) + list(picker_r.FC1.parameters())+list(picker_r.FC2.parameters())+list(picker_r.FC3.parameters())
            # picker =  {'e':picker_e,'r':picker_r}
            # learnable_parameters = [param for name, param in vars(picker).items() if isinstance(param, torch.nn.Parameter) and param.requires_grad]

            self.optimizer_p = torch.optim.Adam(
                [{'params':params_p}], lr = self.adae_config['lr_p'], betas=(0.9, 0.999)
                    ) #用来更新theta的optimizer
            self.kge_lr_scheduler_p = KgeLRScheduler(config, self.optimizer_p)
            self._lr_warmup = self.config.get("train.lr_warmup")
            for group in self.optimizer_p.param_groups:
                group["initial_lr"]=group["lr"]
            self.architect = Architect(self.model,self.optimizer_p, self, self.adae_config)
    # overwrite

    def _prepare(self):
        super()._prepare()
        # select negative sampling implementation
        self._implementation = self.config.check(
            "negative_sampling.implementation", ["triple", "all", "batch", "auto"],
        )
        if self._implementation == "auto":
            max_nr_of_negs = max(self._sampler.num_samples)
            if self._sampler.shared:
                self._implementation = "batch"
            elif max_nr_of_negs <= 30:
                self._implementation = "triple"
            else:
                self._implementation = "batch"
            self.config.set(
                "negative_sampling.implementation", self._implementation, log=True
            )

        self.config.log(
            "Preparing negative sampling training job with "
            "'{}' scoring function ...".format(self._implementation)
        )

        # construct dataloader
        self.num_examples = self.dataset.split(self.train_split).size(0)
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )
    def _get_collate_fun(self):
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

