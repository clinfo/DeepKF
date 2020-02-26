# ==============================================================================
# Load data
# Copyright 2017 Kyoto Univ. Okuno lab. . All Rights Reserved.
# ==============================================================================


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_data(
    config,
    with_shuffle=True,
    with_train_test=True,
    test_flag=False,
    output_dict_flag=True,
):
    time_major = config["time_major"]
    l = None
    if not test_flag:
        x = np.load(config["data_train_npy"])
        if config["mask_train_npy"] is None:
            m = np.ones_like(x)
        else:
            m = np.load(config["mask_train_npy"])
        if config["steps_train_npy"] is None:
            s = [len(x[i]) for i in range(len(x))]
            s = np.array(s)
        else:
            s = np.load(config["steps_train_npy"])
        if config["label_train_npy"] is not None:
            l = np.load(config["label_train_npy"])
    else:
        x = np.load(config["data_test_npy"])
        if config["mask_test_npy"] is None:
            m = np.ones_like(x)
        else:
            m = np.load(config["mask_test_npy"])
        if config["steps_test_npy"] is None:
            s = [len(x[i]) for i in range(len(x))]
            s = np.array(s)
        else:
            s = np.load(config["steps_test_npy"])
        if config["label_test_npy"] is not None:
            l = np.load(config["label_test_npy"])
    if not time_major:
        x = x.transpose((0, 2, 1))
        m = m.transpose((0, 2, 1))
    # train / validatation/ test
    data_num = x.shape[0]
    data_idx = list(range(data_num))
    if with_shuffle:
        np.random.shuffle(data_idx)
    # split train/test
    sep = [0.0, 1.0]
    if with_train_test:
        sep = config["train_test_ratio"]
    prev_idx = 0
    sum_r = 0.0
    sep_idx = []
    for r in sep:
        sum_r += r
        idx = int(data_num * sum_r)
        sep_idx.append([prev_idx, idx])
        prev_idx = idx
    print("#training data:", sep_idx[0][1] - sep_idx[0][0])
    print("#valid data:", sep_idx[1][1] - sep_idx[1][0])

    tr_idx = data_idx[sep_idx[0][0] : sep_idx[0][1]]
    te_idx = data_idx[sep_idx[1][0] : sep_idx[1][1]]
    # storing data
    train_data = dotdict({})
    valid_data = dotdict({})
    tr_x = x[tr_idx]
    te_x = x[te_idx]
    train_data.x = tr_x
    valid_data.x = te_x
    tr_m = m[tr_idx]
    te_m = m[te_idx]
    train_data.m = tr_m
    valid_data.m = te_m
    tr_s = s[tr_idx]
    te_s = s[te_idx]
    train_data.s = tr_s
    valid_data.s = te_s
    if l is not None:
        tr_l = l[tr_idx]
        te_l = l[te_idx]
        train_data.l = tr_l
        valid_data.l = te_l
    train_data.num = tr_x.shape[0]
    valid_data.num = te_x.shape[0]
    train_data.n_steps = tr_x.shape[1]
    valid_data.n_steps = te_x.shape[1]
    train_data.dim = tr_x.shape[2]
    valid_data.dim = tr_x.shape[2]
    return train_data, valid_data
