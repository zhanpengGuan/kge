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


def argparse_bool_type(v):
    "Type for argparse that correctly treats Boolean values"
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def process_meta_command(args, meta_command, fixed_args):
    """Process&update program arguments for meta commands.

    `meta_command` is the name of a special command, which fixes all key-value arguments
    given in `fixed_args` to the specified value. `fxied_args` should contain key
    `command` (for the actual command being run).

    """
    if args.command == meta_command:
        for k, v in fixed_args.items():
            if k != "command" and vars(args)[k] and vars(args)[k] != v:
                raise ValueError(
                    "invalid argument for '{}' command: --{} {}".format(
                        meta_command, k, v
                    )
                )
            vars(args)[k] = v


def create_parser(config, additional_args=[]):
    # define short option names
    short_options = {
        "dataset.name": "-d",
        "job.type": "-j",
        "train.max_epochs": "-e",
        "model": "-m",
    }

    # create parser for config
    parser_conf = argparse.ArgumentParser(add_help=False)
    for key, value in Config.flatten(config.options).items():
        short = short_options.get(key)
        argtype = type(value)
        if argtype == bool:
            argtype = argparse_bool_type
        if short:
            parser_conf.add_argument("--" + key, short, type=argtype)
        else:
            parser_conf.add_argument("--" + key, type=argtype)

    # add additional arguments
    for key in additional_args:
        parser_conf.add_argument(key)

    # add argument to abort on outdated data
    parser_conf.add_argument(
        "--abort-when-cache-outdated",
        action="store_const",
        const=True,
        default=False,
        help="Abort processing when an outdated cached dataset file is found "
        "(see description of `dataset.pickle` configuration key). "
        "Default is to recompute such cache files.",
    )

    # create main parsers and subparsers
    parser = argparse.ArgumentParser("kge")
    subparsers = parser.add_subparsers(title="command", dest="command")
    subparsers.required = True

    # start and its meta-commands
    parser_start = subparsers.add_parser(
        "start", help="Start a new job (create and run it)", parents=[parser_conf]
    )
    parser_create = subparsers.add_parser(
        "create", help="Create a new job (but do not run it)", parents=[parser_conf]
    )
    for p in [parser_start, parser_create]:
        p.add_argument("config", type=str, nargs="?")
        p.add_argument("--folder", "-f", type=str, help="Output folder to use")
        p.add_argument(
            "--run",
            default=p is parser_start,
            type=argparse_bool_type,
            help="Whether to immediately run the created job",
        )

    # resume and its meta-commands
    parser_resume = subparsers.add_parser(
        "resume", help="Resume a prior job", parents=[parser_conf]
    )
    parser_eval = subparsers.add_parser(
        "eval", help="Evaluate the result of a prior job", parents=[parser_conf]
    )
    parser_valid = subparsers.add_parser(
        "valid",
        help="Evaluate the result of a prior job using validation data",
        parents=[parser_conf],
    )
    parser_test = subparsers.add_parser(
        "test",
        help="Evaluate the result of a prior job using test data",
        parents=[parser_conf],
    )
    for p in [parser_resume, parser_eval, parser_valid, parser_test]:
        p.add_argument("config", type=str)
        p.add_argument(
            "--checkpoint",
            type=str,
            help=(
                "Which checkpoint to use: 'default', 'last', 'best', a number "
                "or a file name"
            ),
            default="default",
        )
    add_dump_parsers(subparsers)
    add_package_parser(subparsers)
    return parser


def main():
    # default config
    config = Config() 
    #
    args1 = sys.argv[1:]
    yaml_name = args1[0] if len(args1)>0 else "models/fb15k-237/AdaE_auto.yaml"
    device = args1[1] if len(args1)>1 else "cuda:6"
    # other hyperparameters
    # rank
    debug = True
    if debug:
        rank = True
        if rank:
            dim_list = eval(str(args1[2])) if len(args1)>2 else [64,256]
            # dim = dim_list[-1]
        # fix
        else:
            dim = args1[2] if len(args1)>2 else 256
        lr = args1[3] if len(args1)>3 else "0.5" 
        dropout = args1[4] if len(args1)>4 else "0.1"
        choice_list = eval(str(args1[5])) if len(args1)>5 else [-1]
        t_s = args1[6] if len(args1)>6 else 256
        # auto
        s_u =  args1[7] if len(args1)>7 else 2
        lr_p = args1[8] if len(args1)>8 else 0.01


    # now parse the arguments
    parser = create_parser(config)
    test = False
    # test = True
    if test:
        args, unknown_args = parser.parse_known_args(("test?/data1/gzp/local/fb15k-237/auto/20240107-030555AdaE_auto-auto-cie--0.28--0.1-soft-512-drop-0.5no-gumbel-best").split("?"))
    else:
        args, unknown_args = parser.parse_known_args(("start   "+yaml_name).split())
    # args, unknown_args = parser.parse_known_args(("test?local/experiments/fb15k-237/20231012-055709-AdaE_rank-rank-noshare-[0.999]-[64, 256]-ts-nots256--256-0.5-0.5").split("?"))
   
    
    # If there where unknown args, add them to the parser and reparse. The correctness
    # of these arguments will be checked later.
    if len(unknown_args) > 0:
        parser = create_parser(
            config, filter(lambda a: a.startswith("--"), unknown_args)
        )
        args = parser.parse_args()

    # process meta-commands
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

    # resume command
    if args.command == "resume":
        if os.path.isdir(args.config) and os.path.isfile(args.config + "/config.yaml"):
            args.config += "/config.yaml"
        if not vars(args)["console.quiet"]:
            print("Resuming from configuration {}...".format(args.config))
        config.load(args.config)
        config.folder = os.path.dirname(args.config)
        if not config.folder:
            config.folder = "."
        if not os.path.exists(config.folder):
            raise ValueError(
                "{} is not a valid config file for resuming".format(args.config)
            )

    # overwrite configuration with command line arguments
    for key, value in vars(args).items():
        if key in [
            "command",
            "config",
            "run",
            "folder",
            "checkpoint",
            "abort_when_cache_outdated",
        ]:
            continue
        if value is not None:
            if key == "search.device_pool":
                value = "".join(value).split(",")
            try:
                if isinstance(config.get(key), bool):
                    value = argparse_bool_type(value)
            except KeyError:
                pass
            config.set(key, value)
            if key == "model":
                config._import(value)

    if test:
        config.set('entity_ranking.class_name','EntityRankingJob_freq')

    if args.command == "start":
        # set output folder last str
        last_str="-"
        if debug:
            train_mode = config.get("AdaE_config.train_mode")
            last_str += train_mode
            config.set('job.device', device)
            # config.set('AdaE_config.lr_trans', lr_trans)
            config.set('train.optimizer.default.args.lr',lr)
            # config.set("complex"+'.entity_embedder.dropout', dropout)
            # rank
            if rank:
                config.set('AdaE_config.dim_list', dim_list)
                config.set("multi_lookup_embedder.dim",t_s)
                config.set('AdaE_config.choice_list', choice_list)
                config.set('AdaE_config.s_u', s_u)
                config.set('AdaE_config.lr_p', lr_p)
            else:
                pass
                # config.set("multi_lookup_embedder.dim",dim)
            # print(lr_trans)
            # import time
            # time.sleep(10)

            if train_mode  in  ["auto"]:
                last_str +="-"+ ('cie' if config.get("AdaE_config.cie") else 'nocie' )
                last_str += '-' +str(config.get("AdaE_config.padding"))+ '-'
           
            if train_mode not in  ["original", "fix"]:
                # pass
                # last_str+="-share" if config.get("AdaE_config.share") == True else "-noshare"
                # last_str+="-"+ str(config.get("AdaE_config.choice_list"))+"-"+str(config.get('AdaE_config.dim_list'))
                # # last_str +="-"+str(config.get("AdaE_config.ali_way"))+"-(a)-"
                # last_str+="-"+ str(config.get("train.optimizer.default.args.lr"))
                # last_str +="-soft-"+str(config.get("multi_lookup_embedder.dim"))
                last_str +="-a-"+str(config.get("multi_lookup_embedder.dim"))
                # # last_str+="-drop-"+str(config.get("complex"+'.entity_embedder.dropout'))
                
            if train_mode  in  ["fix"]:
                last_str+="-"+ str(config.get("multi_lookup_embedder.dim"))+"-multilayer-1vsall-"
                last_str+="-"+ str(config.get("train.optimizer.default.args.lr"))
                pass
            if train_mode  in  ["auto"]:
                
                last_str+="-lr_p-"+ str(config.get("AdaE_config.lr_p"))
                last_str+="-"+ str(config.get("AdaE_config.s_u"))
            
        if args.folder is None:  # means: set default
            config_name = os.path.splitext(os.path.basename(args.config))[0]
            config.folder = os.path.join(
                kge_base_dir(),
                "local",
                config.get("dataset.name"),
                config.get("AdaE_config.train_mode"),
                datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+config_name + last_str
            )
            
        else:
            config.folder = args.folder

       # catch errors to log them
    try:
        if args.command == "start" and not config.init_folder():
            raise ValueError("output folder {} exists already".format(config.folder))
        config.log("Using folder: {}".format(config.folder))
        # determine checkpoint to resume (if any)
        if hasattr(args, "checkpoint"):
            checkpoint_file = get_checkpoint_file(config, args.checkpoint)
        # disable processing of outdated cached dataset files globally
        Dataset._abort_when_cache_outdated = args.abort_when_cache_outdated
        # set ra
        seed_from_config(config)
        # let's go
        if args.command == "start" and not args.run:
            config.log("Job created successfully.")
        else:
            # load data
            dataset = Dataset.create(config)
               # AdaE
        # self.rank_e, self.rank_r = self.count_entity_frequency(dataset._triples, dataset._num_entities, dataset._num_relations, adae_config['choice_list'] )
            # let's go
            if args.command == "resume":
                if checkpoint_file is not None:
                    checkpoint = load_checkpoint(
                        checkpoint_file, config.get("job.device")
                    )
                    job = Job.create_from(
                        checkpoint, new_config=config, dataset=dataset
                    )
                else:
                    job = Job.create(config, dataset)
                    job.config.log(
                        "No checkpoint found or specified, starting from scratch..."
                    )
            else:
                job = Job.create(config, dataset)
            # log configuration
            config.log("Configuration:")
            config.log(yaml.dump(config.options), prefix="  ")
            config.log("git commit: {}".format(get_git_revision_short_hash()),
                       prefix="  ")
            job.run()
    except BaseException:
        tb = traceback.format_exc()
        config.log(tb, echo=False)
        raise
if __name__ == "__main__":
    main()