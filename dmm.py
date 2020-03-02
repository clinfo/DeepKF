#
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
import logging

from six.moves import urllib
import tensorflow as tf
import numpy as np
import joblib
import json
import argparse

import dmm.dmm_input as dmm_input

# from dmm_model import inference_by_sample, loss, p_filter, sampleVariationalDist
from dmm.dmm_model import inference, inference_label, loss, p_filter, sampleVariationalDist
from dmm.dmm_model import fivo
from dmm.dmm_model import construct_placeholder, computeEmission, computeVariationalDist
import dmm.hyopt as hy
from dmm.attractor import (
    field,
    potential,
    make_griddata_discrete,
    compute_discrete_transition_mat,
)
# for profiler
from tensorflow.python.client import timeline

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # or map(int, obj)
        return json.JSONEncoder.default(self, obj)


def build_config(config):
    if "result_path" in config:
        path=config["result_path"]
        os.makedirs(path, exist_ok=True)
        config["save_model"]        =path+"/model/model.last.ckpt"
        config["save_model_path"]   =path+"/model"
        config["save_result_filter"]=path+"/filter.jbl"
        config["save_result_test"]  =path+"/test.jbl"
        config["save_result_train"] =path+"/train.jbl"
        config["simulation_path"]   =path+"/sim"
        config["evaluation_output"] =path+"/hyparam.result.json"
        config["load_model"]        =path+"/model/model.last.ckpt"
        config["plot_path"]         =path+"/plot"
        config["log"]         =path+"/log.txt"

def get_default_config():
    config = {}
    # data and network
    # config["dim"]=None
    config["dim"] = 2
    # training
    config["epoch"] = 10
    config["patience"] = 5
    config["batch_size"] = 100
    config["alpha"] = 1.0
    config["learning_rate"] = 1.0e-2
    config["curriculum_alpha"] = False
    config["epoch_interval_save"] = 10  # 100
    config["epoch_interval_print"] = 10  # 100
    config["sampling_tau"] = 10  # 0.1
    config["normal_max_var"] = 5.0  # 1.0
    config["normal_min_var"] = 1.0e-5
    config["zero_dynamics_var"] = 1.0
    config["pfilter_sample_size"] = 10
    config["pfilter_proposal_sample_size"] = 1000
    config["pfilter_save_sample_num"] = 100
    # dataset
    config["train_test_ratio"] = [0.8, 0.2]
    config["data_train_npy"] = None
    config["mask_train_npy"] = None
    config["label_train_npy"] = None
    config["data_test_npy"] = None
    config["mask_test_npy"] = None
    config["label_test_npy"] = None
    config["label_type"] = "multinominal"
    config["task"] = "generative"
    # save/load model
    config["save_model_path"] = None
    config["load_model"] = None
    config["save_result_train"] = None
    config["save_result_test"] = None
    config["save_result_filter"] = None
    # config["state_type"]="discrete"
    config["state_type"] = "normal"
    config["sampling_type"] = "none"
    config["time_major"] = True
    config["steps_train_npy"] = None
    config["steps_test_npy"] = None
    config["sampling_type"] = "normal"
    config["emission_type"] = "normal"
    config["state_type"] = "normal"
    config["dynamics_type"] = "distribution"
    config["pfilter_type"] = "trained_dynamics"
    config["potential_enabled"] = (True,)
    config["potential_grad_transition_enabled"] = (True,)
    config["potential_nn_enabled"] = (False,)
    #
    config["field_grid_num"] = 30
    config["field_grid_dim"] = None
    # generate json
    # fp = open("config.json", "w")
    # json.dump(config, fp, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

    return config


def construct_feed(idx, data, placeholders, alpha, is_train=False):
    feed_dict = {}
    num_potential_points = 100
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]
    n_steps = hy_param["n_steps"]
    batch_size = len(idx)

    dropout_rate = 0.0
    if is_train:
        if "dropout_rate" in hy_param:
            dropout_rate = hy_param["dropout_rate"]
        else:
            dropout_rate = 0.5
    #
    for key, ph in placeholders.items():
        if key == "x":
            feed_dict[ph] = data.x[idx, :, :]
        elif key == "m":
            feed_dict[ph] = data.m[idx, :, :]
        elif key == "s":
            feed_dict[ph] = data.s[idx]
        elif key == "l":
            feed_dict[ph] = data.l[idx, :]
        elif key == "alpha":
            feed_dict[ph] = alpha
        elif key == "vd_eps":
            # eps=np.zeros((batch_size,n_steps,dim))
            if hy_param["state_type"] == "discrete":
                eps = np.random.uniform(
                    1.0e-10, 1.0 - 1.0e-10, (batch_size, n_steps, dim)
                )
                eps = -np.log(-np.log(eps))
            else:
                eps = np.random.standard_normal((batch_size, n_steps, dim))
            feed_dict[ph] = eps
        elif key == "tr_eps":
            # eps=np.zeros((batch_size,n_steps,dim))
            eps = np.random.standard_normal((batch_size, n_steps, dim))
            feed_dict[ph] = eps
        elif key == "potential_points":
            pts = np.random.standard_normal((num_potential_points, dim))
            feed_dict[ph] = pts
        elif key == "dropout_rate":
            feed_dict[ph] = dropout_rate
        elif key == "is_train":
            feed_dict[ph] = is_train
    return feed_dict


def print_variables():
    # print variables
    print("## emission variables")
    vars_em = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="emission_var")
    for v in vars_em:
        print(v.name)
    print("## variational dist. variables")
    vars_vd = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="variational_dist_var"
    )
    for v in vars_vd:
        print(v.name)
    print("## transition variables")
    vars_tr = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="transition_var"
    )
    for v in vars_tr:
        print(v.name)
    print("## potential variables")
    vars_pot = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="potential_var"
    )
    for v in vars_pot:
        print(v.name)
    print("## label variables")
    vars_label = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="label_var")
    for v in vars_label:
        print(v.name)
    return


def compute_alpha(config, i):
    alpha_max = config["alpha"]
    if config["curriculum_alpha"]:
        begin_tau = config["epoch"] * 0.1
        end_tau = config["epoch"] * 0.9
        tau = 100.0
        if i < begin_tau:
            alpha = 0.0
        elif i < end_tau:
            alpha = alpha_max * (1.0 - np.exp(-(i - begin_tau) / tau))
        else:
            alpha = alpha_max
        return alpha
    return alpha_max


class EarlyStopping:
    def __init__(self, config, **kwargs):
        self.prev_validation_cost = None
        self.validation_count = 0
        self.config = config
        self.first = True

    def evaluate_validation(self, validation_cost, info):
        config = self.config
        if (
            self.prev_validation_cost is not None
            and self.prev_validation_cost < validation_cost
        ):
            self.validation_count += 1
            if config["patience"] > 0 and self.validation_count >= config["patience"]:
                self.print_info(info)
                print("[stop] by validation")
                return True
        else:
            self.validation_count = 0
        self.prev_validation_cost = validation_cost
        return False

    def print_info(self, info):
        config = self.config
        epoch = info["epoch"]
        logger = logging.getLogger("logger")
        logger.setLevel(logging.INFO)

        costs_name = info["training_all_costs_name"]
        errors_name = info["training_errors_name"]
        metrics_name = info["training_metrics_name"]

        training_cost = info["training_cost"]
        validation_cost = info["validation_cost"]
        training_errors = info["training_errors"]
        validation_errors = info["validation_errors"]
        training_metrics = info["training_metrics"]
        validation_metrics = info["validation_metrics"]
        training_all_costs = info["training_all_costs"]
        validation_all_costs = info["validation_all_costs"]
        alpha = info["alpha"]
        save_path = info["save_path"]
        if self.first:
            log = "[LOG] epoch, train cost, valid. cost, alpha"
            for name in errors_name:
                log += ", train error(" + name + ")"
            for name in errors_name:
                log += ", valid. error(" + name + ")"
            for name in metrics_name:
                log += ", train (" + name + ")"
            for name in metrics_name:
                log += ", valid. (" + name + ")"
            for name in costs_name:
                log += ", train cost(" + name + ")"
            for name in costs_name:
                log += ", valid. cost(" + name + ")"
            logger.info(log)
            self.first = False
        log = "epoch %d, training cost %g (error=%g), validation cost %g (error=%g)" % (
            epoch,
            training_cost,
            training_errors[0],
            validation_cost,
            validation_errors[0],
        )
        if config["patience"] > 0:
            log += "(EarlyStopping counter=%d/%d)" % (
                self.validation_count,
                self.config["patience"],
            )
        if save_path is None:
            log += "([SAVE] %s) " % (save_path,)
        print(log)
        log = "[LOG] %d, %g, %g, %g" % (epoch, training_cost, validation_cost, alpha)
        for el in training_errors:
            log += ", " + str(el)
        for el in validation_errors:
            log += ", " + str(el)
        for el in training_metrics:
            log += ", " + str(el)
        for el in validation_metrics:
            log += ", " + str(el)
        for el in training_all_costs:
            log += ", " + str(el)
        for el in validation_all_costs:
            log += ", " + str(el)
        logger.info(log)


def compute_cost(
    sess, placeholders, data, data_idx, output_cost, batch_size, alpha, is_train
):
    # initialize costs
    cost = 0.0
    all_costs = None
    all_errors = None
    # compute cost in data
    n_batch = int(np.ceil(data.num * 1.0 / batch_size))
    met = None
    sess.run(tf.local_variables_initializer())
    for j in range(n_batch):
        idx = data_idx[j * batch_size : (j + 1) * batch_size]
        feed_dict = construct_feed(idx, data, placeholders, alpha, is_train=is_train)
        ## computing error/cost
        c, ac, e, met, _ = sess.run(
            [
                output_cost["cost"],
                output_cost["all_costs"],
                output_cost["errors"],
                output_cost["metrics"],
                output_cost["updates"],
            ],
            feed_dict=feed_dict,
        )
        cost += c
        if all_costs is None:
            all_costs = np.array(ac)
        else:
            all_costs += np.array(ac)
        if all_errors is None:
            all_errors = np.array(e)
        else:
            all_errors += np.array(e)
    cost = cost / data.num
    all_errors = all_errors / data.num
    all_costs = all_costs / data.num
    data_info = {
        "cost": cost,
        "errors": all_errors,
        "errors_name": output_cost["errors_name"],
        "metrics": met,
        "metrics_name": output_cost["metrics_name"],
        "all_costs": all_costs,
        "all_costs_name": output_cost["all_costs_name"],
    }
    return data_info


def compute_cost_train_valid(
    sess,
    placeholders,
    train_data,
    valid_data,
    train_idx,
    valid_idx,
    output_cost,
    batch_size,
    alpha,
):
    train_data_info = compute_cost(
        sess,
        placeholders,
        train_data,
        train_idx,
        output_cost,
        batch_size,
        alpha,
        is_train=True,
    )
    valid_data_info = compute_cost(
        sess,
        placeholders,
        valid_data,
        valid_idx,
        output_cost,
        batch_size,
        alpha,
        is_train=False,
    )
    all_info = {}
    for k, v in train_data_info.items():
        all_info["training_" + k] = v
    for k, v in valid_data_info.items():
        all_info["validation_" + k] = v
    return all_info


def compute_result(sess, placeholders, data, data_idx, outputs, batch_size, alpha):
    results = {}
    n_batch = int(np.ceil(data.num * 1.0 / batch_size))
    for j in range(n_batch):
        idx = data_idx[j * batch_size : (j + 1) * batch_size]
        feed_dict = construct_feed(idx, data, placeholders, alpha)
        for k, v in outputs.items():
            if v is not None:
                res = sess.run(v, feed_dict=feed_dict)
                if k in ["z_s"]:
                    if k in results:
                        results[k] = np.concatenate([results[k], res], axis=0)
                    else:
                        results[k] = res
                elif k in [
                    "obs_params",
                    "obs_pred_params",
                    "z_params",
                    "z_pred_params",
                    "label_params",
                ]:
                    if k in results:
                        for i in range(len(res)):
                            results[k][i] = np.concatenate(
                                [results[k][i], res[i]], axis=0
                            )
                    else:
                        if type(res) == tuple:
                            results[k] = list(res)
                        else:
                            results[k] = res
    for k, v in results.items():
        if k in ["z_s"]:
            print(k, v.shape)
        elif k in [
            "obs_params",
            "obs_pred_params",
            "z_params",
            "z_pred_params",
            "label_params",
        ]:
            if len(v) == 1:
                print(k, v[0].shape)
            else:
                print(k, v[0].shape, v[1].shape)
    return results


def get_dim(config, hy_param, data):
    dim_emit = None
    if data is not None:
        dim_emit = data.dim
    elif "dim_emit" in config:
        dim_emit = config["dim_emit"]
    elif "dim_emit" in hy_param:
        dim_emit = hy_param["dim_emit"]
    else:
        dim_emit = 1
    if config["dim"] is None:
        dim = dim_emit
        config["dim"] = dim
    else:
        dim = config["dim"]
    hy_param["dim"] = dim
    hy_param["dim_emit"] = dim_emit
    return dim, dim_emit


def train(sess, config):
    hy_param = hy.get_hyperparameter()
    train_data, valid_data = dmm_input.load_data(
        config, with_shuffle=True, with_train_test=True
    )

    batch_size, n_batch = get_batch_size(config, hy_param, train_data)
    dim, dim_emit = get_dim(config, hy_param, train_data)
    n_steps = train_data.n_steps
    hy_param["n_steps"] = n_steps
    print("train_data_size:", train_data.num)
    print("batch_size     :", batch_size)
    print("n_steps        :", n_steps)
    print("dim_emit       :", dim_emit)

    placeholders = construct_placeholder(config)
    control_params = {"config": config, "placeholders": placeholders}
    # inference
    # outputs=inference_by_sample(n_steps,control_params=control_params)
    outputs = inference(n_steps, control_params=control_params)
    if config["task"] == "label_prediction":
        outputs = inference_label(outputs, n_steps, control_params=control_params)

    # cost
    output_cost = loss(outputs, placeholders["alpha"], control_params=control_params)
    # train_step
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(config["learning_rate"]).minimize(
            output_cost["cost"]
        )
    print_variables()
    saver = tf.train.Saver()
    # initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    train_idx = list(range(train_data.num))
    valid_idx = list(range(valid_data.num))
    ## training
    validation_count = 0
    prev_validation_cost = 0
    alpha = None
    early_stopping = EarlyStopping(config)
    for i in range(config["epoch"]):
        np.random.shuffle(train_idx)
        alpha = compute_alpha(config, i)
        training_info = compute_cost_train_valid(
            sess,
            placeholders,
            train_data,
            valid_data,
            train_idx,
            valid_idx,
            output_cost,
            batch_size,
            alpha,
        )
        # save
        save_path = None
        if i % config["epoch_interval_save"] == 0:
            save_path = saver.save(
                sess, config["save_model_path"] + "/model.%05d.ckpt" % (i)
            )
        # early stopping
        training_info["epoch"] = i
        training_info["alpha"] = alpha
        training_info["save_path"] = save_path
        if i % config["epoch_interval_print"] == 0:
            early_stopping.print_info(training_info)
        if i % 100:
            if early_stopping.evaluate_validation(
                training_info["validation_cost"], training_info
            ):
                break

        # update
        n_batch = int(np.ceil(train_data.num * 1.0 / batch_size))
        for j in range(n_batch):
            idx = train_idx[j * batch_size : (j + 1) * batch_size]
            feed_dict = construct_feed(
                idx, train_data, placeholders, alpha, is_train=True
            )
            train_step.run(feed_dict=feed_dict)

    training_info = compute_cost_train_valid(
        sess,
        placeholders,
        train_data,
        valid_data,
        train_idx,
        valid_idx,
        output_cost,
        batch_size,
        alpha,
    )
    print(
        "[RESULT] training cost %g, validation cost %g, training error %g, validation error %g"
        % (
            training_info["training_cost"],
            training_info["validation_cost"],
            training_info["training_errors"],
            training_info["validation_errors"],
        )
    )
    hy_param["evaluation"] = training_info
    # save hyperparameter
    if config["save_model"] is not None and config["save_model"] != "":
        save_model_path = config["save_model"]
        save_path = saver.save(sess, save_model_path)
        print("[SAVE] %s" % (save_path))
    hy.save_hyperparameter()
    ## save results
    if config["save_result_train"] != "":
        results = compute_result(
            sess, placeholders, train_data, train_idx, outputs, batch_size, alpha
        )
        results["config"] = config
        print("[SAVE] result : ", config["save_result_train"])
        base_path = os.path.dirname(config["save_result_train"])
        os.makedirs(base_path, exist_ok=True)
        joblib.dump(results, config["save_result_train"], compress=3)

        #
        e = (train_data.x - results["obs_params"][0]) ** 2
        #


def infer(sess, config):
    hy_param = hy.get_hyperparameter()
    _, test_data = dmm_input.load_data(
        config, with_shuffle=False, with_train_test=False, test_flag=True
    )
    batch_size, n_batch = get_batch_size(config, hy_param, test_data)
    dim, dim_emit = get_dim(config, hy_param, test_data)
    n_steps = test_data.n_steps
    hy_param["n_steps"] = n_steps
    print("test_data_size:", test_data.num)
    print("batch_size     :", batch_size)
    print("n_steps        :", n_steps)
    print("dim_emit       :", dim_emit)
    alpha = config["alpha"]
    print("alpha          :", alpha)

    placeholders = construct_placeholder(config)
    control_params = {"config": config, "placeholders": placeholders}
    # inference
    outputs = inference(n_steps, control_params)
    if config["task"] == "label_prediction":
        outputs = inference_label(outputs, n_steps, control_params=control_params)
    # cost
    output_cost = loss(outputs, placeholders["alpha"], control_params=control_params)
    # train_step
    saver = tf.train.Saver()
    print_variables()
    print("[LOAD]", config["load_model"])
    saver.restore(sess, config["load_model"])
    test_idx = list(range(test_data.num))
    # check point
    test_info = compute_cost(
        sess,
        placeholders,
        test_data,
        test_idx,
        output_cost,
        batch_size,
        alpha,
        is_train=False,
    )
    print("cost: %g" % (test_info["cost"]))
    print("errors: %g" % (test_info["errors"]))

    ## save results
    if config["save_result_test"] != "":
        results = compute_result(
            sess, placeholders, test_data, test_idx, outputs, batch_size, alpha
        )
        results["config"] = config
        print("[SAVE] result : ", config["save_result_test"])
        base_path = os.path.dirname(config["save_result_test"])
        os.makedirs(base_path, exist_ok=True)
        joblib.dump(results, config["save_result_test"], compress=3)


def filter_discrete_forward(sess, config):
    hy_param = hy.get_hyperparameter()
    _, test_data = dmm_input.load_data(
        config, with_shuffle=False, with_train_test=False, test_flag=True
    )
    batch_size, n_batch = get_batch_size(config, hy_param, test_data)
    dim, dim_emit = get_dim(config, hy_param, test_data)

    n_steps = test_data.n_steps
    hy_param["n_steps"] = n_steps
    z_holder = tf.placeholder(tf.float32, shape=(None, dim))
    z0 = make_griddata_discrete(dim)
    control_params = {"dropout_rate": 0.0, "config": config}
    # inference
    params = computeEmission(
        z_holder, n_steps=1, init_params_flag=True, control_params=control_params
    )

    x_holder = tf.placeholder(tf.float32, shape=(None, n_steps, dim_emit))
    qz = computeVariationalDist(
        x_holder, n_steps, init_params_flag=True, control_params=control_params
    )
    # load
    try:
        saver = tf.train.Saver()
        print("[LOAD] ", config["load_model"])
        saver.restore(sess, config["load_model"])
    except:
        print("[SKIP] Load parameters")
    # computing grid data(state => emission)
    feed_dict = {z_holder: z0}
    x_params = sess.run(params, feed_dict=feed_dict)
    # computing qz (test_x => qz)
    x = test_data.x
    mask = test_data.m
    feed_dict = {x_holder: x}
    out_qz = sess.run(qz, feed_dict=feed_dict)

    # showing mean and variance at each state
    # usually,  num_d == dim
    num_d = x_params[0].shape[0]
    dist_x = []
    for d in range(num_d):
        m = x_params[0][d, 0, :]
        print("##,mean of state=", d, ",".join(map(str, m)))
    for d in range(num_d):
        cov = x_params[1][d, 0, :]
        print("##,var. of state=", d, ",".join(map(str, cov)))
    # compute dist_x
    for d in range(num_d):
        m = x_params[0][d, 0, :]
        cov = x_params[1][d, 0, :]
        diff_x = -(x - m) ** 2 / (2 * cov)
        prob = -1.0 / 2.0 * np.log(2 * np.pi * cov) + diff_x
        # prob: data_num x n_steps x emit_dim
        prob = np.mean(prob * mask, axis=2)
        dist_x.append(prob)
    dist_x = np.array(dist_x)
    dist_x = np.transpose(dist_x, [1, 2, 0])
    # dist: data_num x n_steps x dim(#state)
    dist_x_max = np.zeros_like(dist_x)
    for i in range(dist_x.shape[0]):
        for j in range(dist_x.shape[1]):
            k = np.argmax(dist_x[i, j, :])
            dist_x_max[i, j, k] = 1
    ##
    ## p(x|z)*q(z)
    ## p(x,z)
    dist_qz = out_qz[0].reshape((20, 100, dim))
    dist_pxz = dist_qz * np.exp(dist_x)
    ##
    tr_mat = compute_discrete_transition_mat(sess, config)
    print("original transition matrix")
    print(tr_mat)
    beta = 5.0e-2
    tr_mat = beta * tr_mat + (1.0 - beta) * np.identity(dim)
    print("modified transition matrix")
    print(tr_mat)
    ## viterbi
    prob_viterbi = np.zeros_like(dist_x)
    prob_viterbi[:, :, :] = -np.inf
    path_viterbi = np.zeros_like(dist_x)
    index_viterbi = np.zeros_like(dist_x, dtype=np.int32)
    for d in range(dist_x.shape[0]):
        prob_viterbi[d, 0, :] = dist_pxz[d, 0, :]
        index_viterbi[d, 0, :] = np.argmax(dist_pxz[d, 0, :])
        step = dist_x.shape[1] - 1
        for t in range(step):
            for i in range(dim):
                for j in range(dim):
                    p = 0
                    p += prob_viterbi[d, t, i]
                    p += np.log(dist_pxz[d, t + 1, j])
                    # p+=np.log(dist_qz[d,t+1,j])
                    p += np.log(tr_mat[i, j])
                    if prob_viterbi[d, t + 1, j] < p:
                        prob_viterbi[d, t + 1, j] = p
                        index_viterbi[d, t + 1, j] = i
        ##
        i = np.argmax(prob_viterbi[d, step, :])
        path_viterbi[d, step, i] = 1.0
        for t in range(step):
            j = index_viterbi[d, step - t - 1, i]
            # print(prob_viterbi[d,step-t-1,i])
            path_viterbi[d, step - t - 1, j] = 1.0
            i = j

    ## save results
    if config["save_result_filter"] != "":
        results = {}
        # results["dist"]=dist_x
        results["dist_max"] = dist_x_max
        results["dist_qz"] = dist_qz
        results["dist_pxz"] = dist_pxz
        results["dist_px"] = dist_x
        results["dist_viterbi"] = path_viterbi
        results["tr_mat"] = tr_mat
        print("[SAVE] result : ", config["save_result_filter"])
        joblib.dump(results, config["save_result_filter"], compress=3)


def get_batch_size(config, hy_param, data):
    batch_size = config["batch_size"]
    n_batch = int(data.num / batch_size)
    if n_batch == 0:
        batch_size = data.num
        n_batch = 1
    elif n_batch * batch_size != data.num:
        n_batch += 1
    return batch_size, n_batch


def construct_filter_placeholder(config):
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]
    n_steps = hy_param["n_steps"]
    #
    x_holder = tf.placeholder(tf.float32, shape=(None, dim_emit))
    m_holder = tf.placeholder(tf.float32, shape=(None, dim_emit))
    z_holder = tf.placeholder(tf.float32, shape=(None, n_steps + 1, dim))
    step = tf.placeholder(tf.int32)
    dropout_rate = tf.placeholder(tf.float32)
    is_train = tf.placeholder(tf.bool)
    #
    placeholders = {
        "x": x_holder,
        "z": z_holder,
        "m": m_holder,
        "step": step,
        "dropout_rate": dropout_rate,
        "is_train": is_train,
    }
    return placeholders


def construct_filter_feed(idx, batch_size, step, data, z, placeholders, is_train=False):
    feed_dict = {}
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]
    n_steps = hy_param["n_steps"]
    #sample_size = config["pfilter_sample_size"]
    #proposal_sample_size = config["pfilter_proposal_sample_size"]

    dropout_rate = 0.0
    if is_train:
        if "dropout_rate" in hy_param:
            dropout_rate = hy_param["dropout_rate"]
        else:
            dropout_rate = 0.5
    #
    for key, ph in placeholders.items():
        if key == "x":
            if idx + batch_size > data.num:  # for last
                x = np.zeros((batch_size, dim_emit), dtype=np.float32)
                bs = batch_size - (idx + batch_size - data.num)
                x[:bs, :] = data.x[idx : idx + batch_size, step, :]
            else:
                x = data.x[idx : idx + batch_size, step, :]
                bs = batch_size
            feed_dict[ph] = x
        elif key == "z":
            feed_dict[ph] = z
        elif key == "m":
            if idx + batch_size > data.num:  # for last
                m = np.zeros((batch_size, dim_emit), dtype=np.float32)
                bs = batch_size - (idx + batch_size - data.num)
                m[:bs, :] = data.m[idx : idx + batch_size, step, :]
            else:
                m = data.m[idx : idx + batch_size, step, :]
                bs = batch_size
            feed_dict[ph] = m
        elif key == "dropout_rate":
            feed_dict[ph] = dropout_rate
        elif key == "is_train":
            feed_dict[ph] = is_train
        elif key == "step":
            feed_dict[ph] = step
    return feed_dict, bs


def construct_server_filter_feed(step, x, m, z, placeholders, is_train=False):
    feed_dict = {}
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]
    n_steps = hy_param["n_steps"]
    #sample_size = config["pfilter_sample_size"]
    #proposal_sample_size = config["pfilter_proposal_sample_size"]

    dropout_rate = 0.0
    if is_train:
        if "dropout_rate" in hy_param:
            dropout_rate = hy_param["dropout_rate"]
        else:
            dropout_rate = 0.5
    #
    for key, ph in placeholders.items():
        if key == "x":
            # x=np.zeros((batch_size,dim_emit),dtype=np.float32)
            feed_dict[ph] = x
        elif key == "z":
            feed_dict[ph] = z
        elif key == "m":
            feed_dict[ph] = m
        elif key == "dropout_rate":
            feed_dict[ph] = dropout_rate
        elif key == "is_train":
            feed_dict[ph] = is_train
        elif key == "step":
            feed_dict[ph] = step
    bs = 1
    return feed_dict, bs


def construct_batch_z(idx, batch_size, zs, is_train=False):
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    # zs: sample_size x test_data.num x n_steps+1 x dim
    zz = zs[:, idx : idx + batch_size, :, :]
    sample_size = zz.shape[0]
    s = zz.shape[2]
    bs = zz.shape[1]
    if bs < batch_size:  # for last
        z_temp = np.zeros((sample_size, batch_size, s, dim), dtype=np.float32)
        z_temp[:, :bs, :, :] = zz
    else:
        z_temp = zz
    # zs: (sample_size x batch_size) x n_steps+1 x dim
    return np.reshape(z_temp, [-1, s, dim])


def filtering(sess, config):
    hy_param = hy.get_hyperparameter()
    _, test_data = dmm_input.load_data(
        config, with_shuffle=False, with_train_test=False, test_flag=True
    )
    n_steps = test_data.n_steps
    hy_param["n_steps"] = n_steps
    dim, dim_emit = get_dim(config, hy_param, test_data)
    batch_size, n_batch = get_batch_size(config, hy_param, test_data)

    print(
        "data_size",
        test_data.num,
        "batch_size",
        batch_size,
        ", n_step",
        test_data.n_steps,
        ", dim_emit",
        test_data.dim,
    )
    placeholders = construct_filter_placeholder(config)

    sample_size = config["pfilter_sample_size"]
    proposal_sample_size = config["pfilter_proposal_sample_size"]
    save_sample_num = config["pfilter_save_sample_num"]

    control_params = {"config": config, "placeholders": placeholders}
    # inference
    # z: (batch_size x sample_size) x n_steps x dim
    outputs = p_filter(
        placeholders["x"],
        placeholders["z"],
        placeholders["m"],
        placeholders["step"],
        n_steps + 1,
        None,
        sample_size,
        proposal_sample_size,
        batch_size,
        control_params=control_params,
    )
    # loding model
    print_variables()
    saver = tf.train.Saver()
    print("[LOAD]", config["load_model"])
    saver.restore(sess, config["load_model"])

    zs = np.zeros((sample_size, test_data.num, n_steps + 1, dim), dtype=np.float32)
    # max: proposal_sample_size*sample_size
    sample_idx = list(range(proposal_sample_size * sample_size))
    np.random.shuffle(sample_idx)
    sample_idx = sample_idx[:save_sample_num]
    mus = np.zeros(
        (save_sample_num, test_data.num, n_steps, dim_emit), dtype=np.float32
    )
    errors = np.zeros(
        (save_sample_num, test_data.num, n_steps, dim_emit), dtype=np.float32
    )
    for j in range(n_batch):
        idx = j * batch_size
        print(j, "/", n_batch)
        for step in range(n_steps):
            zs_input = construct_batch_z(idx, batch_size, zs)
            feed_dict, bs = construct_filter_feed(
                idx, batch_size, step, test_data, zs_input, placeholders
            )
            result = sess.run(outputs, feed_dict=feed_dict)
            z = result["sampled_z"]
            # z: sample_size x batch_size x dim
            mu = result["sampled_pred_params"][0]
            zs[:, idx : idx + batch_size, step + 1, :] = z[:, :bs, :]
            mus[:, idx : idx + batch_size, step, :] = mu[sample_idx, :bs, :]
            x = feed_dict[placeholders["x"]]
            errors[:, idx : idx + batch_size, step, :] = (
                mu[sample_idx, :bs, :] - x[:bs, :]
            )
            print("*", end="")
        print("")
    ## save results
    if config["save_result_filter"] != "":
        results = {}
        results["z"] = zs
        results["mu"] = mus
        results["error"] = errors
        print("[SAVE] result : ", config["save_result_filter"])
        joblib.dump(results, config["save_result_filter"], compress=3)


def construct_fivo_placeholder(config):
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]
    n_steps = hy_param["n_steps"]
    #
    x_holder = tf.placeholder(tf.float32, shape=(None, n_steps, dim_emit))
    z0_holder = tf.placeholder(tf.float32, shape=(None, dim_emit))
    dropout_rate = tf.placeholder(tf.float32)
    is_train = tf.placeholder(tf.bool)
    #
    placeholders = {
        "x": x_holder,
        "z": z0_holder,
        "dropout_rate": dropout_rate,
        "is_train": is_train,
    }
    return placeholders


def construct_fivo_feed(data_idx, batch_size, step, data, placeholders, is_train=False):
    feed_dict = {}
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]
    n_steps = hy_param["n_steps"]
    sample_size = hy_param["pfilter_sample_size"]

    dropout_rate = 0.0
    if is_train:
        if "dropout_rate" in hy_param:
            dropout_rate = hy_param["dropout_rate"]
        else:
            dropout_rate = 0.5
    #
    for key, ph in placeholders.items():
        if key == "x":
            idx = data_idx[step * batch_size : (step + 1) * batch_size]
            if len(idx) < batch_size:
                x = np.zeros((batch_size, n_steps, dim), dtype=np.float32)
                x[: len(idx), :, :] = data.x[idx, :, :]
            else:
                x = data.x[idx, :, :]
            feed_dict[ph] = x
        elif key == "z":
            z0 = np.random.normal(0, 1.0, size=(batch_size * sample_size, dim_emit))
            feed_dict[ph] = z0
        elif key == "dropout_rate":
            feed_dict[ph] = dropout_rate
        elif key == "is_train":
            feed_dict[ph] = is_train
    return feed_dict


def train_fivo(sess, config):
    hy_param = hy.get_hyperparameter()
    # _,test_data = dmm_input.load_data(config,with_shuffle=False,with_train_test=False,test_flag=True)
    train_data, valid_data = dmm_input.load_data(
        config, with_shuffle=True, with_train_test=True
    )
    n_steps = train_data.n_steps
    hy_param["n_steps"] = n_steps
    dim, dim_emit = get_dim(config, hy_param, train_data)
    batch_size, n_batch = get_batch_size(config, hy_param, train_data)

    print(
        "data_size",
        train_data.num,
        "batch_size",
        batch_size,
        ", n_step",
        train_data.n_steps,
        ", dim_emit",
        train_data.dim,
    )
    placeholders = construct_fivo_placeholder(config)

    sample_size = config["pfilter_sample_size"]
    proposal_sample_size = config["pfilter_proposal_sample_size"]
    save_sample_num = config["pfilter_save_sample_num"]
    # z0=np.zeros((batch_size*sample_size,dim),dtype=np.float32)
    z0 = np.random.normal(0, 1.0, size=(batch_size * sample_size, dim))
    control_params = {"config": config, "placeholders": placeholders}
    # inference
    # outputs=p_filter(x_holder,z_holder,None,dim,dim_emit,sample_size,batch_size,control_params=control_params)
    output_cost = fivo(
        placeholders["x"],
        placeholders["z"],
        None,
        n_steps,
        sample_size,
        proposal_sample_size,
        batch_size,
        control_params=control_params,
    )
    ##========##
    # train_step
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(config["learning_rate"]).minimize(
            output_cost["cost"]
        )
    print_variables()
    saver = tf.train.Saver()
    if config["profile"]:
        vars_to_train = tf.trainable_variables()
        print(vars_to_train)
        writer = tf.summary.FileWriter("logs", sess.graph)
    # initialize
    init = tf.global_variables_initializer()
    sess.run(init)

    train_idx = list(range(train_data.num))
    valid_idx = list(range(valid_data.num))
    ## training
    validation_count = 0
    prev_validation_cost = 0
    alpha = None
    early_stopping = EarlyStopping(config)
    print(
        "[LOG] epoch, cost,cost(valid.),error,error(valid.),alpha,cost(recons.),cost(temporal),cost(potential),cost(recons.,valid.),cost(temporal,valid),cost(potential,valid)"
    )
    for i in range(config["epoch"]):
        np.random.shuffle(train_idx)
        alpha = compute_alpha(config, i)

        # save
        save_path = None
        if i % config["epoch_interval_save"] == 0:
            save_path = saver.save(
                sess, config["save_model_path"] + "/model.%05d.ckpt" % (i)
            )
        # early stopping

        # update
        n_batch = int(np.ceil(train_data.num * 1.0 / batch_size))
        profiler_start = False
        cost = 0
        for j in range(n_batch):
            print(j, "/", n_batch)
            run_metadata = None
            run_options = None
            if config["profile"] and j == 1 and i == 2:
                profiler_start = True
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            feed_dict = construct_fivo_feed(
                train_idx, batch_size, j, train_data, placeholders, is_train=True
            )
            train_step.run(feed_dict=feed_dict)
            c = sess.run(output_cost["cost"], feed_dict=feed_dict)
            cost += c
            if profiler_start:
                step_stats = run_metadata.step_stats
                tl = timeline.Timeline(step_stats)
                ctf = tl.generate_chrome_trace_format(
                    show_memory=False, show_dataflow=True
                )
                with open("logs/timeline.json", "w") as f:
                    f.write(ctf)
                print("[SAVE] logs/timeline.json")
                profiler_start = False

        print(cost / n_batch)
        ###
        ###
        # result=sess.run(outputs,feed_dict=feed_dict)
    # save hyperparameter
    if config["save_model"] is not None and config["save_model"] != "":
        save_model_path = config["save_model"]
        save_path = saver.save(sess, save_model_path)
        print("[SAVE] %s" % (save_path))
    hy.save_hyperparameter()
    ## save results
    """
	if config["save_result_train"]!="":
		results=compute_result(sess,placeholders,train_data,train_idx,outputs,batch_size,alpha)
		results["config"]=config
		print("[SAVE] result : ",config["save_result_train"])
		base_path = os.path.dirname(config["save_result_train"])
		os.makedirs(base_path,exist_ok=True)
		joblib.dump(results,config["save_result_train"])
		
		#
		e=(train_data.x-results["obs_params"][0])**2
		#
	"""
    return

    ##========##
    # loding model
    print_variables()
    saver = tf.train.Saver()
    print("[LOAD]", config["load_model"])
    saver.restore(sess, config["load_model"])

    feed_dict = {x_holder: test_data.x[0:batch_size, :, :], z0_holder: z0}
    result = sess.run(outputs, feed_dict=feed_dict)

    z = np.reshape(result["sampled_z"], [-1, dim])
    zs = np.zeros((sample_size, test_data.num, n_steps, dim), dtype=np.float32)

    # max: proposal_sample_size*sample_size
    sample_idx = list(range(proposal_sample_size * sample_size))
    np.random.shuffle(sample_idx)
    sample_idx = sample_idx[:save_sample_num]
    mus = np.zeros((sample_size, test_data.num, n_steps, dim_emit), dtype=np.float32)
    errors = np.zeros((sample_size, test_data.num, n_steps, dim_emit), dtype=np.float32)
    for j in range(n_batch):
        idx = j * batch_size
        print(j, "/", n_batch)
        if idx + batch_size > test_data.num:  # for last
            x = np.zeros((batch_size, n_steps, dim), dtype=np.float32)
            bs = batch_size - (idx + batch_size - test_data.num)
            x[:bs, :, :] = test_data.x[idx : idx + batch_size, :, :]
        else:
            x = test_data.x[idx : idx + batch_size, :, :]
            bs = batch_size
        feed_dict = {x_holder: x, z0_holder: z}

        ###
        ###
        result = sess.run(outputs, feed_dict=feed_dict)
        z = result["sampled_z"]
        obs_list = result["sampled_pred_params"]
        ###
        ###
        for step in range(n_steps):
            mu = obs_list[step][0]
            zs[:, idx : idx + batch_size, step, :] = z[step][:, :bs, :]
            print("======")
            # mus:save_sample_num,test_data.num,n_steps,dim_emit
            print(mus.shape)
            print(mu.shape)
            print("======")
            mus[:, idx : idx + batch_size, step, :] = mu[:, :bs, :]
            errors[:, idx : idx + batch_size, step, :] = mu[:, :bs, :] - x[:bs, step, :]
        z = np.reshape(z, [-1, dim])
        print("*", end="")
        print("")
        ##

    ## save results
    if config["save_result_filter"] != "":
        results = {}
        results["z"] = zs
        results["mu"] = mus
        results["error"] = errors
        print("[SAVE] result : ", config["save_result_filter"])
        joblib.dump(results, config["save_result_filter"], compress=3)


def filtering_server(sess, config):
    ## server
    from socket import socket, gethostname, AF_INET, SOCK_DGRAM
    import time
    import datetime

    HOST = gethostname()
    IN_PORT = 34512
    OUT_PORT = 1113
    BUFSIZE = 1024
    print(HOST)
    # ADDR = (gethostbyname(HOST), PORT)
    ADDR = ("127.0.0.1", IN_PORT)
    OUT_ADDR = ("127.0.0.1", OUT_PORT)
    USER = "Server"
    udpServSock = socket(AF_INET, SOCK_DGRAM)  # IPv4/UDP
    udpServSock.bind(ADDR)
    udpServSock.setblocking(0)

    hy_param = hy.get_hyperparameter()
    # _,test_data = dmm_input.load_data(config,with_shuffle=False,with_train_test=False,test_flag=True)
    n_steps = 30  # test_data.n_steps
    hy_param["n_steps"] = n_steps
    dim, dim_emit = get_dim(config, hy_param, None)
    # batch_size,n_batch=get_batch_size(config,hy_param,test_data)
    batch_size = 1
    n_batch = 1
    n = 1

    print(
        "data_size",
        1,
        "batch_size",
        batch_size,
        ", n_step",
        n_steps,
        ", dim ",
        dim,
        ", dim_emit",
        dim_emit,
    )
    placeholders = construct_filter_placeholder(config)

    sample_size = config["pfilter_sample_size"]
    proposal_sample_size = config["pfilter_proposal_sample_size"]
    save_sample_num = config["pfilter_save_sample_num"]

    control_params = {"config": config, "placeholders": placeholders}
    # inference
    # z: (batch_size x sample_size) x n_steps x dim
    outputs = p_filter(
        placeholders["x"],
        placeholders["z"],
        placeholders["m"],
        placeholders["step"],
        n_steps + 1,
        None,
        sample_size,
        proposal_sample_size,
        batch_size,
        control_params=control_params,
    )
    # loding model
    print_variables()
    saver = tf.train.Saver()
    print("[LOAD]", config["load_model"])
    saver.restore(sess, config["load_model"])

    zs = np.zeros((sample_size, n, n_steps + 1, dim), dtype=np.float32)
    # max: proposal_sample_size*sample_size
    sample_idx = list(range(proposal_sample_size * sample_size))
    np.random.shuffle(sample_idx)
    sample_idx = sample_idx[:save_sample_num]
    mus = np.zeros((save_sample_num, n, n_steps, dim_emit), dtype=np.float32)
    errors = np.zeros((save_sample_num, n, n_steps, dim_emit), dtype=np.float32)

    send_cnt = 0
    for j in range(n_batch):
        idx = j * batch_size
        print(j, "/", n_batch)
        # for step in range(n_steps):
        k = 0
        while k < 100000:
            time.sleep(0.01)
            if k < n_steps:
                step = k
            else:
                zs = np.roll(zs, -1, axis=2)
            ###
            print("Waiting for message...")
            data = None
            try:
                data, addr = udpServSock.recvfrom(BUFSIZE)
            except:
                pass
            if data is not None and len(data) > 0:
                print("...received from:", addr)
                #
                a = int.from_bytes(data[0:1], byteorder="little")
                arr = np.frombuffer(data[1:], dtype=np.float32)
                print(a)
                print(arr)
                # label=arr[0]
                x_vec = np.zeros((1, dim_emit), dtype=np.float32)
                x_vec[0, :] = arr[1:]
                m_vec = np.ones((1, dim_emit), dtype=np.float32)
                ###
            else:
                x_vec = np.zeros((1, dim_emit), dtype=np.float32)
                m_vec = np.zeros((1, dim_emit), dtype=np.float32)
            zs_input = construct_batch_z(idx, batch_size, zs)
            feed_dict, bs = construct_server_filter_feed(
                step, x_vec, m_vec, zs_input, placeholders
            )
            result = sess.run(outputs, feed_dict=feed_dict)
            z = result["sampled_z"]
            # z: sample_size x batch_size x dim
            mu = result["sampled_pred_params"][0]
            zs[:, idx : idx + batch_size, step + 1, :] = z[:, :bs, :]
            mus[:, idx : idx + batch_size, step, :] = mu[sample_idx, :bs, :]
            x = feed_dict[placeholders["x"]]
            errors[:, idx : idx + batch_size, step, :] = (
                mu[sample_idx, :bs, :] - x[:bs, :]
            )
            print("*", end="")
            ###
            if data is not None and len(data) > 0:
                data_v = x_vec[0, :] * 10 + 200
                print(">>", data_v)
                data = memoryview(data_v)
                cnt = (send_cnt % 0x100).to_bytes(1, byteorder="little")
                data_type = (0).to_bytes(1, byteorder="little")
                udpServSock.sendto(cnt + data_type + data, OUT_ADDR)
                send_cnt += 1
            #
            data_v = np.mean(z[:, 0, :], axis=0) * 10 + 200
            print(">>", data_v)
            data = memoryview(data_v)
            cnt = (send_cnt % 0x100).to_bytes(1, byteorder="little")
            data_type = (1).to_bytes(1, byteorder="little")
            udpServSock.sendto(cnt + data_type + data, OUT_ADDR)
            send_cnt += 1
            #
            data_v = np.mean(mu[:, 0, :], axis=0) * 10 + 200
            print(">>", data_v)
            data = memoryview(data_v)
            cnt = (send_cnt % 0x100).to_bytes(1, byteorder="little")
            data_type = (2).to_bytes(1, byteorder="little")
            udpServSock.sendto(cnt + data_type + data, OUT_ADDR)
            send_cnt += 1
            #
            k += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help="train/infer")
    parser.add_argument(
        "--config", type=str, default=None, nargs="?", help="config json file"
    )
    parser.add_argument("--no-config", action="store_true", help="use default setting")
    parser.add_argument(
        "--save-config", default=None, nargs="?", help="save config json file"
    )
    parser.add_argument("--model", type=str, default=None, help="model")
    parser.add_argument(
        "--hyperparam",
        type=str,
        default=None,
        nargs="?",
        help="hyperparameter json file",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="cpu mode (calcuration only with cpu)"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="constraint gpus (default: all) (e.g. --gpu 0,2)",
    )
    parser.add_argument("--profile", action="store_true", help="")
    args = parser.parse_args()
    # config
    config = get_default_config()
    if args.config is None:
        if not args.no_config:
            parser.print_help()
            # quit()
    else:
        fp = open(args.config, "r")
        config.update(json.load(fp))
    # if args.hyperparam is not None:
    hy.initialize_hyperparameter(args.hyperparam)
    config.update(hy.get_hyperparameter())
    hy.get_hyperparameter().update(config)
    # build config
    build_config(config)
    # gpu/cpu
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # profile
    config["profile"] = args.profile
    #
    logger = logging.getLogger("logger")
    logger.setLevel(logging.WARN)
    if "log" in config:
        h = logging.FileHandler(filename=config["log"])
        h.setLevel(logging.INFO)
        logger.addHandler(h)

    # setup
    mode_list = args.mode.split(",")
    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    for mode in mode_list:
        with tf.Graph().as_default():
            with tf.Session() as sess:
                # mode
                if mode == "train":
                    train(sess, config)
                elif mode == "infer" or mode == "test":
                    if args.model is not None:
                        config["load_model"] = args.model
                    infer(sess, config)
                elif mode == "filter":
                    if args.model is not None:
                        config["load_model"] = args.model
                    filtering(sess, config)
                elif mode == "filter_discrete":
                    filter_discrete_forward(sess, config)
                elif mode == "train_fivo":
                    train_fivo(sess, config)
                elif mode == "field":
                    field(sess, config)
                elif mode == "potential":
                    potential(sess, config)
                elif args.mode == "filter_server":
                    filtering_server(sess, config=config)

    if args.save_config is not None:
        print("[SAVE] config: ", args.save_config)
        fp = open(args.save_config, "w")
        json.dump(
            config,
            fp,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
            cls=NumPyArangeEncoder,
        )


if __name__ == "__main__":
    main()
