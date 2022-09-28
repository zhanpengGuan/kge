from collections import defaultdict
from copy import deepcopy
import math
import traceback
from kge.util.optimizer import KgeLRScheduler, KgeOptimizer
import time
import torch
import torch.utils.data
from typing import List
from kge.job.trace import format_trace_entry
from kge.job import Job
import kge.job.util
from kge.job.train import TrainingJob, _generate_worker_init_fn

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


class W_TrainingJobKvsAll(TrainingJob):
    """Train with examples consisting of a query and its answers.

    Terminology:
    - Query type: which queries to ask (sp_, s_o, and/or _po), can be configured via
      configuration key `KvsAll.query_type` (which see)
    - Query: a particular query, e.g., (John,marriedTo) of type sp_
    - Labels: list of true answers of a query (e.g., [Jane])
    - Example: a query + its labels, e.g., (John,marriedTo), [Jane]
    """

    from kge.indexing import KvsAllIndex

    def __init__(
        self, config, dataset, parent_job=None, model=None, forward_only=False
    ):
        super().__init__(
            config, dataset, parent_job, model=model, forward_only=forward_only
        )
        self.label_smoothing = config.check_range(
            "KvsAll.label_smoothing", float("-inf"), 1.0, max_inclusive=False
        )
        if self.label_smoothing < 0:
            if config.get("train.auto_correct"):
                config.log(
                    "Setting label_smoothing to 0, "
                    "was set to {}.".format(self.label_smoothing)
                )
                self.label_smoothing = 0
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least 0.".format(self.label_smoothing)
                )
        elif self.label_smoothing > 0 and self.label_smoothing <= (
            1.0 / dataset.num_entities()
        ):
            if config.get("train.auto_correct"):
                # just to be sure it's used correctly
                config.log(
                    "Setting label_smoothing to 1/num_entities = {}, "
                    "was set to {}.".format(
                        1.0 / dataset.num_entities(), self.label_smoothing
                    )
                )
                self.label_smoothing = 1.0 / dataset.num_entities()
            else:
                raise Exception(
                    "Label_smoothing was set to {}, "
                    "should be at least {}.".format(
                        self.label_smoothing, 1.0 / dataset.num_entities()
                    )
                )

        config.log("Initializing 1-to-N training job...")
        self.type_str = "W_KvsAll"
        self.valid_split =  config.get("valid.split")
        self._BP = True
        self.status_2 = None
        self.index = 0
        #self.iter_v = iter(self.loader_v)
        self.index_v = 0
        self.index_0 = 0
        self.optimizer_assumed = None
        self.model_new = None
        if hasattr(self.model,'_base_model'):
            self.model_name = self.model._base_model.model
        else:
            self.model_name = self.model.model
        self.regularize = True if self.config.get("lookup_embedder.regularize")=="lp" else False

        self.model.train()
        self.optimizer_1 = KgeOptimizer.create(config, self.model,type=1)
        self.kge_lr_scheduler = KgeLRScheduler(config, self.optimizer_1)
        self._lr_warmup = self.config.get("train.lr_warmup")
        for group in self.optimizer_1.param_groups:
            group["initial_lr"]=group["lr"]
            
        self.optimizer_2 = KgeOptimizer.create(config, self.model,type=2)
        
        self.kge_lr_scheduler = KgeLRScheduler(config, self.optimizer_2)
        self._lr_warmup = self.config.get("train.lr_warmup")
        for group in self.optimizer_2.param_groups:
            group["initial_lr"]=group["lr"]
        if self.__class__ == W_TrainingJobKvsAll:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        super()._prepare()
        # determine enabled query types
        self.query_types = [
            key
            for key, enabled in self.config.get("W_KvsAll.query_types").items()
            if enabled
        ]

        # corresponding indexes
        self.query_indexes: List[KvsAllIndex] = []

        #' for each query type (ordered as in self.query_types), index right after last
        #' example of that type in the list of all examples (over all query types)
        self.query_last_example = []

        # construct relevant data structures
        self.num_examples = 0
        for query_type in self.query_types:
            index_type = (
                "sp_to_o"
                if query_type == "sp_"
                else ("so_to_p" if query_type == "s_o" else "po_to_s")
            )
            index = self.dataset.index(f"{self.train_split}_{index_type}")
            self.query_indexes.append(index)
            self.num_examples += len(index)
            self.query_last_example.append(self.num_examples)

        # create dataloader
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )
        
        
        # Generate Bi-level optimization dataloader(_t,_v) and they all share one  query_types
        # corresponding indexes
        # loader_t and loader are same dataloaders
        self.loader_t = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )
        
        
        self.query_indexes_v: List[KvsAllIndex] = []

        #' for each query type (ordered as in self.query_types), index right after last
        #' example of that type in the list of all examples (over all query types)
        self.query_last_example_v = []

        # construct relevant data structures
        self.num_examples_v = 0
        for query_type in self.query_types:
            index_type = (
                "sp_to_o"
                if query_type == "sp_"
                else ("so_to_p" if query_type == "s_o" else "po_to_s")
            )
            index = self.dataset.index(f"{self.valid_split}_{index_type}")
            self.query_indexes_v.append(index)
            self.num_examples_v += len(index)
            self.query_last_example_v.append(self.num_examples_v)

        # create dataloader
        self.loader_v = torch.utils.data.DataLoader(
            range(self.num_examples_v),
            collate_fn=self._get_collate_fun_v(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )
        # Iterator of dataloaders,which will be used in run_epoch_v()
        self.iter_t = iter(self.loader_t)
        self.iter_v = iter(self.loader_v)
        self.iter_0 = iter(self.loader)
    def _get_collate_fun(self):
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
    def _get_collate_fun_v(self):
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
                    end = self.query_last_example_v[query_type_index]
                    if example_index < end:
                        example_index -= start
                        num_ones += self.query_indexes_v[query_type_index]._values_offset[
                            example_index + 1
                        ]
                        num_ones -= self.query_indexes_v[query_type_index]._values_offset[
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
                    end = self.query_last_example_v[query_type_index]
                    if example_index < end:
                        example_index -= start
                        query_type_indexes_batch[batch_index] = query_type_index
                        queries = self.query_indexes_v[query_type_index]._keys
                        label_offsets = self.query_indexes_v[
                            query_type_index
                        ]._values_offset
                        labels = self.query_indexes_v[query_type_index]._values
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
    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move labels to GPU for entire batch (else somewhat costly, but this should be
        # reasonably small)
        result.prepare_time -= time.time()
        batch["label_coords"] = batch["label_coords"].to(self.device)
        result.size = len(batch["queries"])
        result.prepare_time += time.time()
    def _process_subbatch_v(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        batch_size = result.size
        # step 2 ,process for model_new
        # prepare
        result.prepare_time -= time.time()
        queries_subbatch = batch["queries"][subbatch_slice].to(self.device)
        subbatch_size = len(queries_subbatch)
        label_coords_batch = batch["label_coords"]
        query_type_indexes_subbatch = batch["query_type_indexes"][subbatch_slice]
        triples = batch['triples']
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
                    scores = self.model_new.score_sp(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                elif query_type == "s_o":
                    scores = self.model_new.score_so(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                else:
                    scores = self.model_new.score_po(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                # note: average on batch_size, not on subbatch_size
                loss_value = (
                    self.loss(scores, labels_for_query_type[query_type],num_negatives= scores.shape[1]-1,triples=triples,model=self.model_new) / batch_size
                )
                result.avg_loss += loss_value.item()
                result.forward_time += time.time()
                result.backward_time -= time.time()
                if not self.is_forward_only:
                    loss_value.backward()
                result.backward_time += time.time()

    
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
                    scores = self.model.score_sp(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                elif query_type == "s_o":
                    scores = self.model.score_so(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )
                else:
                    scores = self.model.score_po(
                        queries_subbatch[examples, 0], queries_subbatch[examples, 1]
                    )

                triples = deepcopy(queries_subbatch[examples])
                # note: average on batch_size, not on subbatch_size
                i = 0
                j = 0
                while i<len(examples) and j<len(label_coords_batch):
                    if examples[i]==label_coords_batch[j][0]:
                        torch.cat((triples[i],torch.tensor([label_coords_batch[j][1]]).to(self.device)),0)
                        j = j + 1
                    else:
                        i = i + 1

                
                loss_value = (
                    self.loss(scores, labels_for_query_type[query_type], num_negatives = scores.shape[1]-1,triples=triples,model=self.model) / batch_size
                )
                result.avg_loss += loss_value.item()
                result.forward_time += time.time()
                result.backward_time -= time.time()
                if not self.is_forward_only:
                    
                    loss_value.backward(retain_graph=True)
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
                    lr=[group["lr"] for group in self.optimizer_1.param_groups],
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
            len_w_update = 0 
            batches = 0
            

            # process batch (preprocessing + forward pass + backward pass on loss)
            done = False
            while not done:
                try:
                            
                    batch_index, batch = self.sampler_0()
                    batch_index_t, batch_t = self.sampler_t()
                    batch_index_v, batch_v = self.sampler_v()
                    if batch_index_t == -1:
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
                            lr=[group["lr"] for group in self.optimizer_1.param_groups],
                        )
                    # run the pre-batch hooks (may update the trace)
                    for f in self.pre_batch_hooks:
                        f(self)


                    # step 1

                    if not self.is_forward_only:
                        self.optimizer_1.zero_grad()
                    batch_result: TrainingJob._ProcessBatchResult = self._process_batch(
                        batch_index, batch
                    )
                    
                    # self.optimizer_1.step()
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
                    if not self.is_forward_only:
                        self.optimizer_1.step()
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

                    # step 2
                    # copy optmizer and model to a new one and one
                    self.model_new =  deepcopy(self.model)
                    self.optimizer_assumed  = KgeOptimizer.create(self.config, self.model_new,type = 1)
                    self.kge_lr_scheduler_assumed = KgeLRScheduler(self.config, self.optimizer_assumed)
                    for group in self.optimizer_assumed.param_groups:
                        group["initial_lr"]=group["lr"]
                    self.optimizer_assumed.load_state_dict(self.optimizer_1.state_dict())

                
            

                    # run the pre-batch hooks (may update the trace)
                    for f in self.pre_batch_hooks:
                        f(self)

                    # try running the batch 
                    
                    if hasattr(self,"_base_model"):
                        self.model_new._base_model.outer = True# with out w loss
                    else:
                        self.model_new.outer = True
                    if not self.is_forward_only:
                        self.optimizer_assumed.zero_grad()
                    self._process_batch_v(
                        batch_index_t, batch_t
                    )
                    
                    self.optimizer_assumed.step()
                    # clone embedding param with require_grad = false 
                    current_emb = []
                    for p in self.optimizer_assumed.param_groups[0]['params']:
                        if len(p.shape)==2 and ((p.shape[0]==self.model._entity_embedder.vocab_size and p.shape[1]==self.model._entity_embedder.dim) or (p.shape[0]==self.model._relation_embedder.vocab_size and p.shape[1]==self.model._relation_embedder.dim)):
                            current_emb.append(p.data.clone())     
                    # clone embedding param_grad        
                    param_grad = []
                    for p in self.optimizer_assumed.param_groups[0]['params']:   
                        if len(p.shape)==2 and ((p.shape[0]==self.model._entity_embedder.vocab_size and p.shape[1]==self.model._entity_embedder.dim) or (p.shape[0]==self.model._relation_embedder.vocab_size and p.shape[1]==self.model._relation_embedder.dim)):
                            param_grad.append(p.grad.clone())
                    
                   
                    if self.config.options['train']['optimizer']['default']['type'] == 'Adam':
                        status = {'step': [v['step']
                                            for i, v in self.optimizer_assumed.state.items() if len(v) > 0 ],
                                'exp_avg': [v['exp_avg'].data.clone()
                                            for i, v in self.optimizer_assumed.state.items() if len(v) > 0 ],
                                'exp_avg_sq': [v['exp_avg_sq'].data.clone()
                                                for i, v in self.optimizer_assumed.state.items() if len(v) > 0 ]}
                        # filter other params and save embedding param
                        pop_index=[]
                        for key in status.keys():
                            if key=='exp_avg':
                                for i,p in enumerate(status[key]):
                                    if not (len(p.shape)==2 and 
                                    ((p.shape[0]==self.model._entity_embedder.vocab_size and p.shape[1]==self.model._entity_embedder.dim) or 
                                    (p.shape[0]==self.model._relation_embedder.vocab_size and p.shape[1]==self.model._relation_embedder.dim))):
                                        pop_index.append(i)
                        status_filter ={
                                'step': [v
                                            for i, v in enumerate(status['step']) if i not in pop_index ],
                                'exp_avg':  [v
                                            for i, v in enumerate(status['exp_avg']) if i not in pop_index  ],
                                'exp_avg_sq':[v
                                            for i, v in enumerate(status['exp_avg_sq']) if i not in pop_index  ]}  
                        if hasattr(self,"_base_model"):
                            self.model._base_model.assumed_emb = self.get_next_emb_adam(param_grad,current_emb,status_filter,self.regularize,self.model_name)
                        else:
                            self.model.assumed_emb = self.get_next_emb_adam(param_grad,current_emb,status_filter,self.regularize,self.model_name)
                    elif self.config.options['train']['optimizer']['default']['type'] == 'Adagrad':
                        status = {'step': [v['step']
                                            for i, v in self.optimizer_assumed.state.items() if len(v) > 0 ],
                                'sum': [v['sum'].data.clone()
                                            for i, v in self.optimizer_assumed.state.items() if len(v) > 0 ],
                                }
                        # filter other params and save embedding param
                        pop_index=[]
                        for key in status.keys():
                            if key!='step':
                                for i,p in enumerate(status[key]):
                                    if not (len(p.shape)==2 and 
                                    ((p.shape[0]==self.model._entity_embedder.vocab_size and p.shape[1]==self.model._entity_embedder.dim) or 
                                    (p.shape[0]==self.model._relation_embedder.vocab_size and p.shape[1]==self.model._relation_embedder.dim))):
                                        pop_index.append(i)
                        status_filter ={
                                'step': [v
                                            for i, v in enumerate(status['step']) if i not in pop_index ],
                                'sum':  [v
                                            for i, v in enumerate(status['sum']) if i not in pop_index  ],
                        }         
                            
                        if hasattr(self,"_base_model"):
                            self.model._base_model.assumed_emb = self.get_next_emb_adagrad(param_grad,current_emb,status_filter,self.regularize,self.model_name)
                        else:
                            self.model.assumed_emb = self.get_next_emb_adagrad(param_grad,current_emb,status_filter,self.regularize,self.model_name)

                    # get next adam update emb
                    
                    if hasattr(self,"_base_model"):
                        self.model._base_model.outer = True
                        self.model._base_model.W_update = True
                    else:
                        self.model.outer = True
                        self.model.W_update = True
                    self.optimizer_2.zero_grad()
                    self._process_batch(
                        batch_index_v, batch_v
                    )
                    # penalties_torch = self.model.penalty(
                    #     epoch=self.epoch,
                    #     batch_index=batch_index_v,
                    #     num_batches=len(self.loader_v),
                    #     batch=batch_v,
                    #     W_update=True
                    # )
                    # # backward pass on penalties 
                    # for index, (penalty_key, penalty_value_torch) in enumerate(penalties_torch):
                    #     if not self.is_forward_only:
                    #         penalty_value_torch.backward()
                    # torch.nn.utils.clip_grad_norm_(self.optimizer_2.param_groups[0]['params'], 100)
                    self.optimizer_2.step()
                    if hasattr(self,"_base_model"):
                        self.model._base_model._weight_entity.data.clamp_(0.01,self.model._base_model.clip)
                        self.model._base_model._weight_relation.data.clamp_(0.01,self.model._base_model.clip)
                        self.model._base_model.assumed_emb = None
                        self.model._base_model.W_update = False
                        self.model._base_model.outer = False
                    else:
                        self.model._weight_entity.data.clamp_(0.01,self.model.clip)
                        self.model._weight_relation.data.clamp_(0.01,self.model.clip)
                        self.model.assumed_emb = None
                        self.model.W_update = False
                        self.model.outer = False


                    
                    # dump Weight params
                    import os

                    path = 'local/data/'+self.config.folder.split('/')[-1]
                    isExists=os.path.exists(path) #判断路径是否存在，存在则返回true
                    if not isExists:
                    #如果不存在则创建目录
                    #创建目录操作函数
                        os.makedirs(path) 
                    else:
                        pass
                    torch.save(self.optimizer_2.param_groups[0]['params'][0].data.cpu().numpy(),'local/data/'+self.config.folder.split('/')[-1]+'/entity_Weight.pth')
                    torch.save(self.optimizer_2.param_groups[0]['params'][1].data.cpu().numpy(),'local/data/'+self.config.folder.split('/')[-1]+'/relation_Weight.pth')

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
    def sampler_0(self):
        try:
            self.index_0 += 1
            return self.index_0,next(self.iter_0)
        except StopIteration:
            self.iter_0 = iter(self.loader)
            self.index_0 = 0
            return -1,0
    def get_next_emb_adam(self, emb_grad_no_reg, curr_emb, curr_mf_optim_status,regularize,model_name):
        assert emb_grad_no_reg is not None
        assert curr_mf_optim_status is not None
        lr = self.optimizer_1.defaults['lr']
        eps = self.optimizer_1.defaults['eps']
        beta1, beta2 = self.optimizer_1.defaults['betas'][0], self.optimizer_1.defaults['betas'][1]

        user_emb_cur, item_emb_cur = curr_emb

        t, _ = curr_mf_optim_status['step']
        s, r = curr_mf_optim_status['exp_avg'], curr_mf_optim_status['exp_avg_sq']
        s_u, s_i = s
        r_u, r_i = r
        # gradients of assumed regularized update
        user_emb_grad_no_reg, item_emb_grad_no_reg = emb_grad_no_reg
        if regularize:
            e_emb_grad = user_emb_grad_no_reg*self.optimizer_2.param_groups[0]['params'][0].reshape(-1,1) +  self.config.get(model_name+".entity_embedder.regularize_weight") * user_emb_cur
            r_emb_grad = item_emb_grad_no_reg*self.optimizer_2.param_groups[0]['params'][1].reshape(-1,1) +  self.config.get(model_name+"W_transe.relation_embedder.regularize_weight") * item_emb_cur
        else:
            e_emb_grad = user_emb_grad_no_reg*self.optimizer_2.param_groups[0]['params'][0].reshape(-1,1)
            r_emb_grad = item_emb_grad_no_reg*self.optimizer_2.param_groups[0]['params'][1].reshape(-1,1)



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
    def get_next_emb_adagrad(self, emb_grad_no_reg, curr_emb, curr_mf_optim_status,regularize,model_name):
        assert emb_grad_no_reg is not None
        assert curr_mf_optim_status is not None
        lr = self.optimizer_1.defaults['lr']
        eps = self.optimizer_1.defaults['eps']
        

        user_emb_cur, item_emb_cur = curr_emb

        t, _ = curr_mf_optim_status['step']
        s= curr_mf_optim_status['sum']
        s_u, s_i = s
        
        # gradients of assumed regularized update
        user_emb_grad_no_reg, item_emb_grad_no_reg = emb_grad_no_reg
        if regularize:
            e_emb_grad = user_emb_grad_no_reg*self.optimizer_2.param_groups[0]['params'][0].reshape(-1,1) +  self.config.get(model_name+".entity_embedder.regularize_weight") * user_emb_cur
            r_emb_grad = item_emb_grad_no_reg*self.optimizer_2.param_groups[0]['params'][1].reshape(-1,1) +  self.config.get(model_name+".relation_embedder.regularize_weight") * item_emb_cur
        else:
            e_emb_grad = user_emb_grad_no_reg*self.optimizer_2.param_groups[0]['params'][0].reshape(-1,1)
            r_emb_grad = item_emb_grad_no_reg*self.optimizer_2.param_groups[0]['params'][1].reshape(-1,1)



        s_u = s_u.addcmul(1,e_emb_grad,e_emb_grad) 
        s_i = s_i.addcmul(1,r_emb_grad,r_emb_grad)
        # r_u = beta2 * r_u + (1 - beta2) * user_emb_grad * user_emb_grad
        # r_i = beta2 * r_i + (1 - beta2) * item_emb_grad * item_emb_grad
        
        denom_u = s_u.add(eps).sqrt() # Note: avoid gradient near zero 0.5 x^(-1/2)
        denom_i = s_i.add(eps).sqrt()
       
        step_size = lr 
        e_emb = user_emb_cur.addcdiv(-step_size, e_emb_grad, denom_u)
        r_emb = item_emb_cur.addcdiv(-step_size, r_emb_grad, denom_i)
        return [e_emb, r_emb] 