
#!/usr/bin/env python
import datetime
import argparse
import os
import sys
import traceback
import yaml

from kge import Dataset
from kge import Config
from kge.job import Job
from kge.misc import get_git_revision_short_hash, kge_base_dir, is_number
from kge.util.dump import add_dump_parsers, dump
from kge.util.io import get_checkpoint_file, load_checkpoint
from kge.util.package import package_model, add_package_parser
from kge.util.seed import seed_from_config
from kge.cli import process_meta_command,create_parser,argparse_bool_type




if __name__=="__main__":
    config = Config() 
    #
    args1 = sys.argv[1:]
    yaml_name = args1[0] if len(args1)>0 else "models/WNRR18/AdaE_rank.yaml"
    parser = create_parser(config)
    args, unknown_args = parser.parse_known_args(("start   "+yaml_name).split())

    process_meta_command(args, "create", {"command": "start", "run": False})
    process_meta_command(args, "eval", {"command": "resume", "job.type": "eval"})
    process_meta_command(
        args, "test", {"command": "resume", "job.type": "eval", "eval.split": "test"}
    )
    process_meta_command(
        args, "valid", {"command": "resume", "job.type": "eval", "eval.split": "valid"}
    )
    # dump command
    if args.command == "dump":
        dump(args)
        exit()

    # package command
    if args.command == "package":
        package_model(args)
        exit()

    # start command
    if args.command == "start":
        # use toy config file if no config given
        if args.config is None:
            args.config = kge_base_dir() + "/" + "examples/toy-complex-train.yaml"
            print(
                "WARNING: No configuration specified; using " + args.config,
                file=sys.stderr,
            )

        if not vars(args)["console.quiet"]:
            print("Loading configuration {}...".format(args.config))
        config.load(args.config)
    dataset = Dataset.create(config)
    rank_e, rank_r, count_e, pivot = dataset.count_entity_frequency(dataset._triples['train'], dataset._num_entities, dataset._num_relations, [0.2], dataset.folder)
    rank_e, rank_r, count_e, pivot = dataset.count_entity_frequency(dataset._triples['train'], dataset._num_entities, dataset._num_relations, [0.2], dataset.folder)

