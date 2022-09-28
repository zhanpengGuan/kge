from ast import Dict
from collections import defaultdict
from copy import deepcopy
from kge.util.optimizer import KgeLRScheduler, KgeOptimizer
import traceback
from kge.job.trace import format_trace_entry
import math
import time
from typing import Any
import torch
import torch.utils.data

from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.util import KgeSampler
from kge.model.transe import TransEScorer

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


class W_TrainingJobNegativeSampling(TrainingJob):
    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        self._sampler = KgeSampler.create(config, "W_negative_sampling", dataset)
        self.type_str = "W_negative_sampling"
        self.valid_split =  config.get("valid.split")
        self._BP = True
        self.status_2 = None
        self.index = 0
        #self.iter_v = iter(self.loader_v)
        self.index_v = 0
        self.optimizer_assumed = None
        self.model_new = None

        if self.__class__ == W_TrainingJobNegativeSampling:
            for f in Job.job_created_hooks:
                f(self)

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

        # generate Bi-level dataloader
        self.num_examples_t = self.dataset.split(self.train_split).size(0)
        self.loader_t = torch.utils.data.DataLoader(
            range(self.num_examples_t),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )
        self.num_examples_v = self.dataset.split(self.valid_split).size(0)
        self.loader_v = torch.utils.data.DataLoader(
            range(self.num_examples_v),
            collate_fn=self._get_collate_fun_v(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )
        self.iter_t = iter(self.loader_t)
        self.iter_v = iter(self.loader_v)

   
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
            return {"triples": triples, "negative_samples": negative_samples}

        return collate
    def _get_collate_fun_v(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n,3], ),
            - negative_samples (list of tensors of shape [n,num_samples]; 3 elements
              in order S,P,O)
            """

            triples = self.dataset.split(self.valid_split)[batch, :].long()
            # labels = torch.zeros((len(batch), self._sampler.num_negatives_total + 1))
            # labels[:, 0] = 1
            # labels = labels.view(-1)

            negative_samples = list()
            for slot in [S, P, O]:
                negative_samples.append(self._sampler.sample(triples, slot))
            return {"triples": triples, "negative_samples": negative_samples}

        return collate

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
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

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        batch_size = result.size

        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice]
        batch_negative_samples = batch["negative_samples"]
        subbatch_size = len(triples)
        result.prepare_time += time.time()
        labels = batch["labels"]  # reuse b/w subbatches
        i_num = 1
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
            if self._BP:
                pass
            # compute the scores
            result.forward_time -= time.time()
            scores = torch.empty((subbatch_size, num_samples + 1), device=self.device)
            scores[:, 0] = self.model.score_spo(
                triples[:, S], triples[:, P], triples[:, O], direction=SLOT_STR[slot],
            )
            result.forward_time += time.time()
            scores[:, 1:] = batch_negative_samples[slot].score(
                self.model, indexes=subbatch_slice
            )
            result.forward_time += batch_negative_samples[slot].forward_time
            result.prepare_time += batch_negative_samples[slot].prepare_time

            # compute loss for slot in subbatch (concluding the forward pass)
            result.forward_time -= time.time()
            loss_value_torch = (
                self.loss(scores, labels[slot], num_negatives=num_samples,triples = triples,model = self.model) / batch_size
            )
            result.avg_loss += loss_value_torch.item()
            result.forward_time += time.time()

            # backward pass for this slot in the subbatch
            result.backward_time -= time.time()
            if not self.is_forward_only:
                if i_num<len([S, P, O]):
                    loss_value_torch.backward(retain_graph=True)
                else:
                    loss_value_torch.backward()
            i_num += 1
            result.backward_time += time.time() 
    def _process_batch_v(self, batch_index, batch):
        "Breaks a batch into subbatches and processes them in turn."
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
            self._process_subbatch_v(batch_index, batch, subbatch_slice, result)

        return result
    def _process_subbatch_v(
            self,
            batch_index,
            batch,
            subbatch_slice,
            result: TrainingJob._ProcessBatchResult,
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
            if self._BP:
                pass
            # compute the scores
            result.forward_time -= time.time()
            scores = torch.empty((subbatch_size, num_samples + 1), device=self.device)
            scores[:, 0] = self.model_new.score_spo(
                triples[:, S], triples[:, P], triples[:, O], direction=SLOT_STR[slot],
            )
            result.forward_time += time.time()
            scores[:, 1:] = batch_negative_samples[slot].score(
                self.model_new, indexes=subbatch_slice
            )
            result.forward_time += batch_negative_samples[slot].forward_time
            result.prepare_time += batch_negative_samples[slot].prepare_time

            # compute loss for slot in subbatch (concluding the forward pass)
            result.forward_time -= time.time()
            loss_value_torch = (
                self.loss(scores, labels[slot], num_negatives=num_samples,triples = triples,model = self.model_new) / batch_size
            )
            result.avg_loss += loss_value_torch.item()
            result.forward_time += time.time()

            # backward pass for this slot in the subbatch
            result.backward_time -= time.time()
            if not self.is_forward_only:
                if self.model_new._base_model.outer == True:
                    loss_value_torch.backward()
                else:
                    loss_value_torch.backward()
                
            result.backward_time += time.time()
    def run_epoch_v(self):
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
            done1 = False

            # copy optmizer and model to a new one and one
            
            self.model_new =  deepcopy(self.model)
            self.optimizer_assumed  = KgeOptimizer.create(self.config, self.model_new)
            self.kge_lr_scheduler_assumed = KgeLRScheduler(self.config, self.optimizer_assumed)
            for group in self.optimizer_assumed.param_groups:
                group["initial_lr"]=group["lr"]
            
            self.optimizer_assumed.load_state_dict(self.optimizer.state_dict())

            # restore weight moment if self.status_2 is not none
            if self.status_2 is not None:
                num_i = 2
                for i, v in self.optimizer.state.items() :
                    #emb.weight shape is 2D，change the moment to zero to stop the its update 
                    if len(v) > 0 and len(v['exp_avg'].shape) == 1:
                        v['exp_avg'] = self.status_2['exp_avg'][num_i]
                        v['exp_avg_sq'] = self.status_2['exp_avg_sq'][num_i]
                        num_i += 1
           
            



            # stay optimizer embedding params
            num_i = 0
            status_1 = {'step': [deepcopy(v['step'])
                                for i, v in self.optimizer.state.items() if len(v) > 0 ],
                                'exp_avg': [deepcopy(v['exp_avg'])
                                            for i, v in self.optimizer.state.items() if len(v) > 0 ],
                                'exp_avg_sq': [deepcopy(v['exp_avg_sq'])
                                                for i, v in self.optimizer.state.items() if len(v) > 0 ]}
            for i, v in self.optimizer.state.items() :
                #emb.weight shape is 2D，change the moment to zero to stop the its update 
                if len(v) > 0 and len(v['exp_avg'].shape) == 2:
                    v['exp_avg'] = torch.zeros(v['exp_avg'].shape).cuda()
                    v['exp_avg_sq'] = torch.zeros(v['exp_avg_sq'].shape).cuda()
            # process each batch
            len_w_update = 0 
            batches = 0
            while not done1:
                batch_index, batch = self.sampler_t()
                batch_index_v, batch_v = self.sampler_v()
                if batch_index == -1:
                    break
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
                        lr=[group["lr"] for group in self.optimizer_assumed.param_groups],
                    )

                # run the pre-batch hooks (may update the trace)
                for f in self.pre_batch_hooks:
                    f(self)

                # process batch (preprocessing + forward pass + backward pass on loss)
                done = False
                while not done:
                    try:
                        # try running the batch
                        
                        
                        self.model_new._base_model.outer = True# with out w loss
                        
                        if not self.is_forward_only:
                            self.optimizer_assumed.zero_grad()
                        batch_result: TrainingJob._ProcessBatchResult = self._process_batch_v(
                            batch_index, batch
                        )
                        
                        self.optimizer_assumed.step()
                        # clone param with require_grad = false
                        current_emb = []
                        i = 0
                        for p in self.optimizer_assumed.param_groups[0]['params']:
                            i += 1 
                            if i > 2 :
                                current_emb.append(p.data.clone())     
                        # clone param_grad  
                        i = 0                 
                        param_grad = []
                        for p in self.optimizer_assumed.param_groups[0]['params']:
                            i += 1 
                            if i > 2 :
                                param_grad.append(p.grad.clone())
                        status = {'step': [v['step']
                                            for i, v in self.optimizer_assumed.state.items() if len(v) > 0 ],
                                'exp_avg': [v['exp_avg'].data.clone()
                                            for i, v in self.optimizer_assumed.state.items() if len(v) > 0 ],
                                'exp_avg_sq': [v['exp_avg_sq'].data.clone()
                                                for i, v in self.optimizer_assumed.state.items() if len(v) > 0 ]}
                        for key in status.keys():
                            if len(status[key])>2:
                                status[key].pop() 
                                status[key].pop() 
                        
                        
                        
                        # get next adam update emb
                       
                        self.model._base_model.assumed_emb = self.get_next_emb_adam(param_grad,current_emb,status)
                        self.model._base_model.outer = True
                        self.model._base_model.W_update = True
                        self.optimizer.zero_grad()
                        batch_result: TrainingJob._ProcessBatchResult = self._process_batch(
                            batch_index_v, batch_v
                        )
                        if self.optimizer.param_groups[0]['params'][0].grad is not None:
                            if  torch.equal(self.optimizer.param_groups[0]['params'][2].grad,torch.zeros(14541,128).cuda()):
                                len_w_update += len(self.optimizer.param_groups[0]['params'][0].grad.nonzero())
                                batches += 1
                        self.optimizer.step()
                        self.model._base_model._weight_entity.data.clamp_(0.01,self.model._base_model.clip)
                        self.model._base_model._weight_relation.data.clamp_(0.01,self.model._base_model.clip)
                        self.model._base_model.assumed_emb = None
                        self.model._base_model.W_update = False
                        self.model._base_model.outer = False

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
                    batch=batch,
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

            # restore the emb.weight moment to adam
            num_i = 0
            for i, v in self.optimizer.state.items() :
                #emb.weight shape is 2D，change the moment to zero to stop the its update 
                if len(v) > 0 and len(v['exp_avg'].shape) == 2:
                    v['exp_avg'] = status_1['exp_avg'][num_i]
                    v['exp_avg_sq'] = status_1['exp_avg_sq'][num_i]
                    num_i += 1
            #stay weight moment to status
            self.status_2 = {'step': [deepcopy(v['step'])
                                for i, v in self.optimizer.state.items() if len(v) > 0 ],
                                'exp_avg': [deepcopy(v['exp_avg'])
                                            for i, v in self.optimizer.state.items() if len(v) > 0 ],
                                'exp_avg_sq': [deepcopy(v['exp_avg_sq'])
                                                for i, v in self.optimizer.state.items() if len(v) > 0 ]}
            for i, v in self.optimizer.state.items() :
                #weight shape is 1D，change the moment to zero to stop the its update 
                if len(v) > 0 and len(v['exp_avg'].shape) == 1:
                    v['exp_avg'] = torch.zeros(v['exp_avg'].shape).cuda()
                    v['exp_avg_sq'] = torch.zeros(v['exp_avg_sq'].shape).cuda()
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
            print(len_w_update*1.0/batches)
            return trace_entry
    def sampler_t(self):
        try:
            self.index += 1
            return self.index,next(self.iter_t)
        except StopIteration:
            self.iter_t = iter(self.loader_t)
            self.index = 0
            return -1,0
    def sampler_v(self):
        try:
            self.index_v += 1
            return self.index_v,next(self.iter_v)
        except StopIteration:
            self.iter_v = iter(self.loader_v)
            self.index_v = 0
            return 0,next(self.iter_v)
    def get_next_emb_adam(self, emb_grad_no_reg, curr_emb, curr_mf_optim_status):
        assert emb_grad_no_reg is not None
        assert curr_mf_optim_status is not None
        lr = self.optimizer.defaults['lr']
        eps = self.optimizer.defaults['eps']
        beta1, beta2 = self.optimizer.defaults['betas'][0], self.optimizer.defaults['betas'][1]

        user_emb_cur, item_emb_cur = curr_emb

        t, _ = curr_mf_optim_status['step']
        s, r = curr_mf_optim_status['exp_avg'], curr_mf_optim_status['exp_avg_sq']
        s_u, s_i = s
        r_u, r_i = r
        # gradients of assumed regularized update
        user_emb_grad_no_reg, item_emb_grad_no_reg = emb_grad_no_reg
        e_emb_grad = user_emb_grad_no_reg*self.optimizer.param_groups[0]['params'][0].reshape(-1,1)
        r_emb_grad = item_emb_grad_no_reg*self.optimizer.param_groups[0]['params'][1].reshape(-1,1)

        s_u = beta1 * s_u + (1 - beta1) * e_emb_grad
        s_i = beta1 * s_i + (1 - beta1) * r_emb_grad
        # r_u = beta2 * r_u + (1 - beta2) * user_emb_grad * user_emb_grad
        # r_i = beta2 * r_i + (1 - beta2) * item_emb_grad * item_emb_grad
        r_u = r_u.mul(beta2).addcmul(1 - beta2, e_emb_grad, e_emb_grad)
        r_i = r_i.mul(beta2).addcmul(1 - beta2, r_emb_grad, r_emb_grad)
        denom_u = r_u.add(eps).sqrt() # Note: avoid gradient near zero 0.5 x^(-1/2)
        denom_i = r_i.add(eps).sqrt()
        bias_correction1 = (1 - beta1 ** t)
        bias_correction2 = (1 - beta2 ** t)
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        e_emb = user_emb_cur.addcdiv(-step_size, s_u, denom_u)
        r_emb = item_emb_cur.addcdiv(-step_size, s_i, denom_i)
        return [e_emb, r_emb]