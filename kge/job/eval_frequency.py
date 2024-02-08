import math
import time
import sys

import torch
import kge.job
from kge.job import EvaluationJob, Job
from kge import Config, Dataset
from kge.job import EntityRankingJob
from collections import defaultdict
from kge.job.trace import format_trace_entry

class EntityRankingJob_freq(EntityRankingJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        
    def _prepare(self):
        super()._prepare()
        """Construct all indexes needed to run."""

        # create data and precompute indexes
        self.triples = self.dataset.split(self.config.get("eval.split"))
        for split in self.filter_splits:
            self.dataset.index(f"{split}_sp_to_o")
            self.dataset.index(f"{split}_po_to_s")
        if "test" not in self.filter_splits and self.filter_with_test:
            self.dataset.index("test_sp_to_o")
            self.dataset.index("test_po_to_s")

        # and data loader
        self.rank_e, self.rank_r, self.count_e, self.pivot = self.dataset.count_entity_frequency(self.dataset._triples['train'], self.dataset._num_entities, self.dataset._num_relations, [0.2], self.dataset.folder)
        self.rank_e, self.rank_r = self.rank_e.to(self.device), self.rank_r.to(self.device)
        self.rank = self.rank_e
        self.count_high,self.count_low,self.count_mixed = 0,0,0
        self.triples_high,self.triples_low,self.triples_mixed,self.triples_example = [],[],[],[]
        for head, relation, tail in self.triples:
            if head == 3084 or tail==3084:
                self.triples_example.append([head,relation,tail])
            else:
                if self.rank[head] == self.rank[tail] and self.rank[head]==1:
                    self.triples_high.append([head,relation,tail])
                    self.count_high += 1
                elif self.rank[head] == self.rank[tail] and self.rank[head]==0:
                    self.triples_low.append([head,relation,tail])
                    self.count_low +=1
                else:
                    self.triples_mixed.append([head,relation,tail])
                    self.count_mixed+=1
        self.triples_high = torch.tensor(self.triples_high)
        self.triples_low = torch.tensor(self.triples_low)
        self.triples_mixed = torch.tensor(self.triples_mixed)
        self.triples_example = torch.tensor(self.triples_example)
        
        self.loader = torch.utils.data.DataLoader(
            self.triples,
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )
        
 

    @torch.no_grad()
    def _evaluate(self):
        num_entities = self.dataset.num_entities()

        # we also filter with test data if requested
        filter_with_test = "test" not in self.filter_splits and self.filter_with_test

        # which rankings to compute (DO NOT REORDER; code assumes the order given here)
        rankings = (
            ["_raw", "_filt", "_filt_test"] if filter_with_test else ["_raw", "_filt"]
        )

        # dictionary that maps entry of rankings to a sparse tensor containing the
        # true labels for this option
        labels_for_ranking = defaultdict(lambda: None)

        # Initiliaze dictionaries that hold the overall histogram of ranks of true
        # answers. These histograms are used to compute relevant metrics. The dictionary
        # entry with key 'all' collects the overall statistics and is the default.
        hists = dict()
        hists_filt = dict()
        hists_filt_test = dict()

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type="entity_ranking",
            scope="epoch",
            split=self.eval_split,
            filter_splits=self.filter_splits,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples),
        )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        # let's go
        epoch_time = -time.time()
        for batch_number, batch_coords in enumerate(self.loader):
            # create initial batch trace (yet incomplete)
            self.current_trace["batch"] = dict(
                type="entity_ranking",
                scope="batch",
                split=self.eval_split,
                filter_splits=self.filter_splits,
                epoch=self.epoch,
                batch=batch_number,
                size=len(batch_coords[0]),
                batches=len(self.loader),
            )

            # run the pre-batch hooks (may update the trace)
            for f in self.pre_batch_hooks:
                f(self)

            # construct a sparse label tensor of shape batch_size x 2*num_entities
            # entries are either 0 (false) or infinity (true)
            # TODO add timing information
            batch = batch_coords[0].to(self.device)
            s, p, o = batch[:, 0], batch[:, 1], batch[:, 2]
            label_coords = batch_coords[1].to(self.device)
            if filter_with_test:
                test_label_coords = batch_coords[2].to(self.device)
                # create sparse labels tensor
                test_labels = kge.job.util.coord_to_sparse_tensor(
                    len(batch),
                    2 * num_entities,
                    test_label_coords,
                    self.device,
                    float("Inf"),
                )
                labels_for_ranking["_filt_test"] = test_labels

            # create sparse labels tensor
            labels = kge.job.util.coord_to_sparse_tensor(
                len(batch), 2 * num_entities, label_coords, self.device, float("Inf")
            )
            labels_for_ranking["_filt"] = labels

            # compute true scores beforehand, since we can't get them from a chunked
            # score table
            # o_true_scores = self.model.score_spo(s, p, o, "o").view(-1)
            # s_true_scores = self.model.score_spo(s, p, o, "s").view(-1)
            # scoring with spo vs sp and po can lead to slight differences for ties
            # due to floating point issues.
            # We use score_sp and score_po to stay consistent with scoring used for
            # further evaluation.
            unique_o, unique_o_inverse = torch.unique(o, return_inverse=True)
            o_true_scores = torch.gather(
                self.model.score_sp(s, p, unique_o),
                1,
                unique_o_inverse.view(-1, 1),
            ).view(-1)
            unique_s, unique_s_inverse = torch.unique(s, return_inverse=True)
            s_true_scores = torch.gather(
                self.model.score_po(p, o, unique_s),
                1,
                unique_s_inverse.view(-1, 1),
            ).view(-1)

            # default dictionary storing rank and num_ties for each key in rankings
            # as list of len 2: [rank, num_ties]
            ranks_and_ties_for_ranking = defaultdict(
                lambda: [
                    torch.zeros(s.size(0), dtype=torch.long, device=self.device),
                    torch.zeros(s.size(0), dtype=torch.long, device=self.device),
                ]
            )

            # calculate scores in chunks to not have the complete score matrix in memory
            # a chunk here represents a range of entity_values to score against
            if self.config.get("entity_ranking.chunk_size") > -1:
                chunk_size = self.config.get("entity_ranking.chunk_size")
            else:
                chunk_size = self.dataset.num_entities()

            # process chunk by chunk
            for chunk_number in range(math.ceil(num_entities / chunk_size)):
                chunk_start = chunk_size * chunk_number
                chunk_end = min(chunk_size * (chunk_number + 1), num_entities)

                # compute scores of chunk
                scores = self.model.score_sp_po(
                    s, p, o, torch.arange(chunk_start, chunk_end, device=self.device)
                )
                scores_sp = scores[:, : chunk_end - chunk_start]
                scores_po = scores[:, chunk_end - chunk_start :]

                # replace the precomputed true_scores with the ones occurring in the
                # scores matrix to avoid floating point issues
                s_in_chunk_mask = (chunk_start <= s) & (s < chunk_end)
                o_in_chunk_mask = (chunk_start <= o) & (o < chunk_end)
                o_in_chunk = (o[o_in_chunk_mask] - chunk_start).long()
                s_in_chunk = (s[s_in_chunk_mask] - chunk_start).long()

                # check that scoring is consistent up to configured tolerance
                # if this is not the case, evaluation metrics may be artificially inflated
                close_check = torch.allclose(
                    scores_sp[o_in_chunk_mask, o_in_chunk],
                    o_true_scores[o_in_chunk_mask],
                    rtol=self.tie_rtol,
                    atol=self.tie_atol,
                )
                close_check &= torch.allclose(
                    scores_po[s_in_chunk_mask, s_in_chunk],
                    s_true_scores[s_in_chunk_mask],
                    rtol=self.tie_rtol,
                    atol=self.tie_atol,
                )
                if not close_check:
                    diff_a = torch.abs(
                        scores_sp[o_in_chunk_mask, o_in_chunk]
                        - o_true_scores[o_in_chunk_mask]
                    )
                    diff_b = torch.abs(
                        scores_po[s_in_chunk_mask, s_in_chunk]
                        - s_true_scores[s_in_chunk_mask]
                    )
                    diff_all = torch.cat((diff_a, diff_b))
                    self.config.log(
                        f"Tie-handling: mean difference between scores was: {diff_all.mean()}."
                    )
                    self.config.log(
                        f"Tie-handling: max difference between scores was: {diff_all.max()}."
                    )
                    error_message = "Error in tie-handling. The scores assigned to a triple by the SPO and SP_/_PO scoring implementations were not 'equal' given the configured tolerances. Verify the model's scoring implementations or consider increasing tie-handling tolerances."
                    if self.config.get("entity_ranking.tie_handling.warn_only"):
                        print(error_message, file=sys.stderr)
                    else:
                        raise ValueError(error_message)

                # now compute the rankings (assumes order: None, _filt, _filt_test)
                for ranking in rankings:
                    if labels_for_ranking[ranking] is None:
                        labels_chunk = None
                    else:
                        # densify the needed part of the sparse labels tensor
                        labels_chunk = self._densify_chunk_of_labels(
                            labels_for_ranking[ranking], chunk_start, chunk_end
                        )

                        # remove current example from labels
                        labels_chunk[o_in_chunk_mask, o_in_chunk] = 0
                        labels_chunk[
                            s_in_chunk_mask, s_in_chunk + (chunk_end - chunk_start)
                        ] = 0

                    # compute partial ranking and filter the scores (sets scores of true
                    # labels to infinity)
                    (
                        s_rank_chunk,
                        s_num_ties_chunk,
                        o_rank_chunk,
                        o_num_ties_chunk,
                        scores_sp_filt,
                        scores_po_filt,
                    ) = self._filter_and_rank(
                        scores_sp, scores_po, labels_chunk, o_true_scores, s_true_scores
                    )

                    # from now on, use filtered scores
                    scores_sp = scores_sp_filt
                    scores_po = scores_po_filt

                    # update rankings
                    ranks_and_ties_for_ranking["s" + ranking][0] += s_rank_chunk
                    ranks_and_ties_for_ranking["s" + ranking][1] += s_num_ties_chunk
                    ranks_and_ties_for_ranking["o" + ranking][0] += o_rank_chunk
                    ranks_and_ties_for_ranking["o" + ranking][1] += o_num_ties_chunk

                # we are done with the chunk

            # We are done with all chunks; calculate final ranks from counts
            s_ranks = self._get_ranks(
                ranks_and_ties_for_ranking["s_raw"][0],
                ranks_and_ties_for_ranking["s_raw"][1],
            )
            o_ranks = self._get_ranks(
                ranks_and_ties_for_ranking["o_raw"][0],
                ranks_and_ties_for_ranking["o_raw"][1],
            )
            s_ranks_filt = self._get_ranks(
                ranks_and_ties_for_ranking["s_filt"][0],
                ranks_and_ties_for_ranking["s_filt"][1],
            )
            o_ranks_filt = self._get_ranks(
                ranks_and_ties_for_ranking["o_filt"][0],
                ranks_and_ties_for_ranking["o_filt"][1],
            )

            # Update the histograms of of raw ranks and filtered ranks
            batch_hists = dict()
            batch_hists_filt = dict()
            for f in self.hist_hooks:
                f(batch_hists, s, p, o, s_ranks, o_ranks, job=self)
                f(batch_hists_filt, s, p, o, s_ranks_filt, o_ranks_filt, job=self)

            # and the same for filtered_with_test ranks
            if filter_with_test:
                batch_hists_filt_test = dict()
                s_ranks_filt_test = self._get_ranks(
                    ranks_and_ties_for_ranking["s_filt_test"][0],
                    ranks_and_ties_for_ranking["s_filt_test"][1],
                )
                o_ranks_filt_test = self._get_ranks(
                    ranks_and_ties_for_ranking["o_filt_test"][0],
                    ranks_and_ties_for_ranking["o_filt_test"][1],
                )
                for f in self.hist_hooks:
                    f(
                        batch_hists_filt_test,
                        s,
                        p,
                        o,
                        s_ranks_filt_test,
                        o_ranks_filt_test,
                        job=self,
                    )

            # optionally: trace ranks of each example
            if self.trace_examples:
                entry = {
                    "type": "entity_ranking",
                    "scope": "example",
                    "split": self.eval_split,
                    "filter_splits": self.filter_splits,
                    "size": len(batch),
                    "batches": len(self.loader),
                    "epoch": self.epoch,
                }
                for i in range(len(batch)):
                    entry["batch"] = i
                    entry["s"], entry["p"], entry["o"] = (
                        s[i].item(),
                        p[i].item(),
                        o[i].item(),
                    )
                    if filter_with_test:
                        entry["rank_filtered_with_test"] = (
                            o_ranks_filt_test[i].item() + 1
                        )
                    self.trace(
                        event="example_rank",
                        task="sp",
                        rank=o_ranks[i].item() + 1,
                        rank_filtered=o_ranks_filt[i].item() + 1,
                        **entry,
                    )
                    if filter_with_test:
                        entry["rank_filtered_with_test"] = (
                            s_ranks_filt_test[i].item() + 1
                        )
                    self.trace(
                        event="example_rank",
                        task="po",
                        rank=s_ranks[i].item() + 1,
                        rank_filtered=s_ranks_filt[i].item() + 1,
                        **entry,
                    )

            # Compute the batch metrics for the full histogram (key "all")
            metrics = self._compute_metrics(batch_hists["all"])
            metrics.update(
                self._compute_metrics(batch_hists_filt["all"], suffix="_filtered")
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        batch_hists_filt_test["all"], suffix="_filtered_with_test"
                    )
                )

            # update batch trace with the results
            self.current_trace["batch"].update(metrics)

            # run the post-batch hooks (may modify the trace)
            for f in self.post_batch_hooks:
                f(self)

            # output, then clear trace
            if self.trace_batch:
                self.trace(**self.current_trace["batch"])
            self.current_trace["batch"] = None

            # output batch information to console
            self.config.print(
                (
                    "\r"  # go back
                    + "{}  batch:{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}, mrr (filt.): {:4.3f} ({:4.3f}), "
                    + "hits@1: {:4.3f} ({:4.3f}), "
                    + "hits@{}: {:4.3f} ({:4.3f})"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_number,
                    len(self.loader) - 1,
                    metrics["mean_reciprocal_rank"],
                    metrics["mean_reciprocal_rank_filtered"],
                    metrics["hits_at_1"],
                    metrics["hits_at_1_filtered"],
                    self.hits_at_k_s[-1],
                    metrics["hits_at_{}".format(self.hits_at_k_s[-1])],
                    metrics["hits_at_{}_filtered".format(self.hits_at_k_s[-1])],
                ),
                end="",
                flush=True,
            )

            # merge batch histograms into global histograms
            def merge_hist(target_hists, source_hists):
                for key, hist in source_hists.items():
                    if key in target_hists:
                        target_hists[key] = target_hists[key] + hist
                    else:
                        target_hists[key] = hist

            merge_hist(hists, batch_hists)
            merge_hist(hists_filt, batch_hists_filt)
            if filter_with_test:
                merge_hist(hists_filt_test, batch_hists_filt_test)

        # we are done; compute final metrics
        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back
        for key, hist in hists.items():
            name = "_" + key if key != "all" else ""
            metrics.update(self._compute_metrics(hists[key], suffix=name))
            metrics.update(
                self._compute_metrics(hists_filt[key], suffix="_filtered" + name)
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        hists_filt_test[key], suffix="_filtered_with_test" + name
                    )
                )
        epoch_time += time.time()

        # update trace with results
        self.current_trace["epoch"].update(
            dict(epoch_time=epoch_time, event="eval_completed", **metrics,)
        )

    def _run(self):
        was_training = self.model.training
        self.model.eval()
        # for write in folder
        self.config.log_folder = None
        # gzp
        self.config.log(
            "Evaluating on "
            + self.eval_split
            + " data (epoch {})...".format(self.epoch)
        )

        freq = ['high','low','mixed','all']
        # freq = ['example']
        avg = 0
        for i in freq:
            if i=='high':
                self.loader = torch.utils.data.DataLoader(
                    self.triples_high,
                    collate_fn=self._collate,
                    shuffle=False,
                    batch_size=self.batch_size,
                    num_workers=self.config.get("eval.num_workers"),
                    pin_memory=self.config.get("eval.pin_memory"),
                )
                self._evaluate()
            elif i=='low':
                self.loader = torch.utils.data.DataLoader(
                    self.triples_low,
                    collate_fn=self._collate,
                    shuffle=False,
                    batch_size=self.batch_size,
                    num_workers=self.config.get("eval.num_workers"),
                    pin_memory=self.config.get("eval.pin_memory"),
                )
                self._evaluate()
            elif i=="mixed":
                self.loader = torch.utils.data.DataLoader(
                    self.triples_mixed,
                    collate_fn=self._collate,
                    shuffle=False,
                    batch_size=self.batch_size,
                    num_workers=self.config.get("eval.num_workers"),
                    pin_memory=self.config.get("eval.pin_memory"),
                )
                self._evaluate()
            elif i=='all':
                self.loader = torch.utils.data.DataLoader(
                    self.triples,
                    collate_fn=self._collate,
                    shuffle=False,
                    batch_size=self.batch_size,
                    num_workers=self.config.get("eval.num_workers"),
                    pin_memory=self.config.get("eval.pin_memory"),
                )
                self._evaluate()
            else:
                self.loader = torch.utils.data.DataLoader(
                    self.triples_example,
                    collate_fn=self._collate,
                    shuffle=False,
                    batch_size=self.batch_size,
                    num_workers=self.config.get("eval.num_workers"),
                    pin_memory=self.config.get("eval.pin_memory"),
                )
                self._evaluate()

            # if validation metric is not present, try to compute it
            metric_name = self.config.get("valid.metric")
            if metric_name not in self.current_trace["epoch"]:
                self.current_trace["epoch"][metric_name] = eval(
                    self.config.get("valid.metric_expr"),
                    None,
                    dict(config=self.config, **self.current_trace["epoch"]),
                )

            # run hooks (may modify trace)
            for f in self.post_epoch_hooks:
                f(self)

            # output the trace, then clear it
            trace_entry = self.trace(**self.current_trace["epoch"], echo=False, log=True)

            # 检测平均值
            if i=="high":
                avg += trace_entry['mean_reciprocal_rank_filtered']*self.count_high
                self.config.log("-"*18+"\n"+"high MRR_filt:{}\n".format(trace_entry['mean_reciprocal_rank_filtered'])+"-"*18+"\n")
            elif i=='low':
                avg +=  trace_entry['mean_reciprocal_rank_filtered']*self.count_low
                self.config.log("-"*18+"\n"+"low MRR_filt:{}\n".format(trace_entry['mean_reciprocal_rank_filtered'])+"-"*18+"\n")
            elif i=='mixed':
                avg +=  trace_entry['mean_reciprocal_rank_filtered']*self.count_mixed
                self.config.log("-"*18+"\n"+"mixed MRR_filt:{}\n".format(trace_entry['mean_reciprocal_rank_filtered'])+"-"*18+"\n")
            elif i=='all':
                self.config.log("-"*18+"\n"+"all MRR_filt:{}".format(avg/(self.count_high+self.count_low+self.count_mixed))+"\n"+"-"*18+"\n")
            else:
                self.config.log("-"*18+"\n"+"example MRR_filt:{}\n".format(trace_entry['mean_reciprocal_rank_filtered'])+"-"*18+"\n")
            self.config.log(
                format_trace_entry("eval_epoch", trace_entry, self.config), prefix="  "
            )
            self.current_trace["epoch"] = None

        # reset model and return metrics
        if was_training:
            self.model.train()
        self.config.log("Test ratio:high:{},low:{},mixed:{}\n".format(self.count_high,self.count_low,self.count_mixed))
        self.config.log("Finished evaluating on " + self.eval_split + " split.")

        return trace_entry