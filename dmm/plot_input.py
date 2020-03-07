import numpy as np
import joblib
import json
import sys
import os
import argparse
from dmm.dmm_input import load_data
from dmm.dmm import get_default_config, build_config


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def get_default_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="one/all")
    parser.add_argument(
        "--config", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument(
        "--hyperparam", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument(
        "--input", type=str, default=None, nargs="?", help="input file .jbl",
    )
    parser.add_argument(
        "--info_key",
        type=str,
        default="pid_list_test",
        nargs="?",
        help="the key of the info file for data list, e.g., pid_list_test/pid_list_train",
    )
    parser.add_argument(
        "--out_dir", type=str, default="plot_test", help="output directory for images"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="data index of plotting target (only test mode)",
    )
    parser.add_argument(
        "--limit_all",
        type=int,
        default=None,
        help="the number of output images (all mode)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="data index of plotting target (only test mode)",
    )
    return parser


def make_default_info(x):
    info = {}
    info["pid_list_test"] = ["data" + str(i) for i in range(x.shape[0])]
    info["attr_emit_list"] = ["attr" + str(i) for i in range(x.shape[2])]
    return info


def get_param(config, key, default_value):
    if config is not None and key in config:
        return config[key]
    return default_value


def load_config(args):
    fp = open(args.config, "r")
    config = get_default_config()
    config.update(json.load(fp))
    build_config(config)
    if args.hyperparam == "":
        fp = open(args.hyperparam, "r")
        config.update(json.load(fp))
    return config


def load_plot_data(args, config=None):
    if config is None:
        config = load_config(args)

    test_flag = False
    print("mode:", args.mode)
    if args.mode == "train":
        result_key = "save_result_train"
    elif args.mode == "infer":
        result_key = "save_result_test"
        test_flag = True
    elif args.mode == "filter":
        result_key = "save_result_filter"
        test_flag = True
    elif args.mode == "data":
        result_key = None

    _, data = load_data(
        config, with_shuffle=False, with_train_test=False, test_flag=test_flag
    )

    if args.input:
        filename_result = args.input
        print("[LOAD]:", filename_result)
        result = joblib.load(filename_result)
        result_data = dotdict(result)
    elif result_key:
        print("result:", result_key)
        filename_result = config[result_key]
        print("[LOAD]:", filename_result)
        result = joblib.load(filename_result)
        result_data = dotdict(result)
    else:
        result_data = dotdict()
    pid_key = args.info_key
    out_dir = args.out_dir
    out_dir = get_param(config, "plot_path", out_dir)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result_data.out_dir = out_dir
    result_data.pid_key = pid_key

    filename_info = get_param(config, "data_info_json", None)
    if filename_info:
        print("[LOAD]:", filename_info)
        fp = open(filename_info, "r")
        result_data.info = json.load(fp)
    else:
        result_data.info = make_default_info(data.x)
    return data, result_data
