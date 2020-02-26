#
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf
import numpy as np
import hyopt as hy

import layers

# FLAGS = tf.app.flags.FLAGS


def construct_placeholder(config):
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]
    n_steps = hy_param["n_steps"]
    #
    x_holder = tf.placeholder(tf.float32, shape=(None, n_steps, dim_emit))
    m_holder = tf.placeholder(tf.float32, shape=(None, n_steps, dim_emit))
    s_holder = tf.placeholder(tf.int32, shape=(None,))
    vd_eps_holder = tf.placeholder(tf.float32, shape=(None, n_steps, dim))
    tr_eps_holder = tf.placeholder(tf.float32, shape=(None, n_steps, dim))
    potential_points_holder = tf.placeholder(tf.float32, shape=(None, dim))
    alpha_holder = tf.placeholder(tf.float32)
    dropout_rate = tf.placeholder(tf.float32)
    is_train = tf.placeholder(tf.bool)
    #
    placeholders = {
        "x": x_holder,
        "m": m_holder,
        "s": s_holder,
        "potential_points": potential_points_holder,
        "alpha": alpha_holder,
        "vd_eps": vd_eps_holder,
        "tr_eps": tr_eps_holder,
        "dropout_rate": dropout_rate,
        "is_train": is_train,
    }
    if config["task"] == "label_prediction":
        placeholders["l"] = tf.placeholder(tf.int32, shape=(None, n_steps))
    return placeholders


def build_nn(
    x, dim_input, n_steps, hyparam_name, name, init_params_flag, control_params
):
    """
	Retruns the layer of the neural networks. 
	Parameters
	----------
		x :

		dim_input :

		hyparm_name :

	Returns
	-------
		layer :

		layer_dim :

	"""
    wd_bias = None
    wd_w = 0.1
    layer = x
    layer_dim = dim_input
    res_layer = None
    hy_param = hy.get_hyperparameter()
    for i, hy_layer in enumerate(hy_param[hyparam_name]):
        layer_dim_out = layer_dim
        # print(">>>",layer_dim)
        if "dim_output" in hy_layer:
            layer_dim_out = hy_layer["dim_output"]
            # print(">>>",layer_dim,"=>",layer_dim_out)

        if hy_layer["name"] == "fc":
            with tf.variable_scope(name + "_fc" + str(i)) as scope:
                layer = layers.fc_layer(
                    "vd_fc" + str(i),
                    layer,
                    layer_dim,
                    layer_dim_out,
                    wd_w,
                    wd_bias,
                    activate=tf.sigmoid,
                    init_params_flag=init_params_flag,
                )
                layer_dim = layer_dim_out
        elif hy_layer["name"] == "fc_res_start":
            res_layer = layer
        elif hy_layer["name"] == "do":
            dropout_rate = control_params["placeholders"]["dropout_rate"]
            layer = layers.dropout_layer(layer, dropout_rate)
            layer_dim = layer_dim_out
        elif hy_layer["name"] == "fc_res":
            with tf.variable_scope(name + "_fc_res" + str(i)) as scope:
                layer = layers.fc_layer(
                    name + "_fc_res" + str(i),
                    layer,
                    layer_dim,
                    layer_dim_out,
                    wd_w,
                    wd_bias,
                    activate=None,
                    init_params_flag=init_params_flag,
                )
                layer = res_layer + layer
                layer = tf.sigmoid(layer)
                layer_dim = layer_dim_out
        elif hy_layer["name"] == "fc_bn":
            is_train = control_params["placeholders"]["is_train"]
            with tf.variable_scope(name + "_fc_bn" + str(i)) as scope:
                layer = layers.fc_layer(
                    name + "_fcbn" + str(i),
                    layer,
                    layer_dim,
                    layer_dim_out,
                    wd_w,
                    wd_bias,
                    activate=tf.sigmoid,
                    with_bn=True,
                    init_params_flag=init_params_flag,
                    is_train=is_train,
                )
                layer_dim = layer_dim_out
        elif hy_layer["name"] == "cnn":
            with tf.variable_scope(name + "_cnn" + str(i)) as scope:
                layer = tf.reshape(layer, [-1, n_steps, layer_dim])
                layer = tf.layers.conv1d(
                    layer,
                    layer_dim_out,
                    1,
                    padding="SAME",
                    reuse=(not init_params_flag),
                    name="conv" + str(i),
                )
                layer = tf.reshape(layer, [-1, layer_dim_out])
                layer_dim = layer_dim_out
        elif hy_layer["name"] == "lstm":
            with tf.variable_scope(name + "_lstm" + str(i)) as scope:
                layer = tf.reshape(layer, [-1, n_steps, layer_dim])
                layer = layers.lstm_layer(
                    layer, n_steps, layer_dim_out, init_params_flag=init_params_flag
                )
                layer = tf.reshape(layer, [-1, layer_dim_out])
                layer_dim = layer_dim_out
        else:
            assert "unknown layer:" + hy_layer["name"]
    return layer, layer_dim


def sample_normal(params, eps):
    """
	Parameters
	----------
		params : list
			mu : 
				mean
			cov : 
				covariance
		eps : 
			tiny value
	Returns
	-------

	"""
    mu = params[0]
    cov = params[1]
    if eps is not None:
        return mu + tf.sqrt(cov) * eps
    else:
        return mu


def sampleState(qz, control_params, sample_shape=()):
    """
	Sampling z_s from the distribution q(z)
	Parameters
	----------
		qz :  

	Returns
	-------
		z_s :

	"""
    sttype = control_params["config"]["state_type"]
    stype = control_params["config"]["sampling_type"]
    if sttype == "discrete" or sttype == "discrete_tr":
        if stype == "none":
            z_s = qz[0]
        elif stype == "gambel-max" or stype == "gumbel-max":
            eps = control_params["placeholders"]["vd_eps"]
            g = eps
            # g=-tf.log(-tf.log(eps))
            logpi = tf.log(qz[0] + 1.0e-10)
            dist = tf.contrib.distributions.OneHotCategorical(logits=(logpi + g))
            z_s = tf.cast(dist.sample(sample_shape), tf.float32)
        elif stype == "gambel-softmax" or stype == "gumbel-softmax":
            tau = control_params["config"]["sampling_tau"]
            eps = control_params["placeholders"]["vd_eps"]
            g = eps
            # g=-tf.log(-tf.log(eps))
            logpi = tf.log(qz[0] + 1.0e-5)
            z_s = tf.nn.softmax((logpi + g) / tau)
        elif stype == "naive":
            dist = tf.contrib.distributions.OneHotCategorical(probs=qz[0])
            z_s = tf.cast(dist.sample(sample_shape), tf.float32)
        else:
            raise Exception("[Error] unknown sampling type")
    elif sttype == "normal":
        if stype == "none":
            z_s = qz[0]
        elif stype == "normal" and "vd_eps" in control_params["placeholders"]:
            eps = control_params["placeholders"]["vd_eps"]
            z_s = sample_normal(qz, eps)
        elif stype == "normal":
            dist = tf.contrib.distributions.Normal(qz[0], qz[1])
            z_s = tf.cast(dist.sample(sample_shape), tf.float32)
        else:
            raise Exception("[Error] unknown sampling type")
    else:
        raise Exception("[Error] unknown state type")
    return z_s


def sampleVariationalDist(x, n_steps, init_params_flag=True, control_params=None):
    """
	Parameters
	----------
		x :
			the observed value x_{1:t}.
	Returns
	-------
		qz :
			the parameters of the variational distribution q(z|x)
		qs :
			the state z_s which is sampled from the variational distribution q(z|x)
	"""
    qz = computeVariationalDist(x, n_steps, init_params_flag, control_params)
    qs = sampleState(qz, control_params)
    return qs, qz


# return parameters for q(z): list of tensors: batch_size x n_steps x dim
def computeVariationalDist(x, n_steps, init_params_flag=True, control_params=None):
    """
	Return parameters for q(z|x_{1:t}).
	Parameters
	----------
	x : 
		bs x t x dim_emit
		or (bs x t) x dim_emit

	Returns
	-------
	params : list
		layer_z : 
			bs x T x dim
		layer_mu : 
			bs x T x dim
		layer_cov : 
			bs x T x dim
	"""
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]
    wd_bias = None
    wd_w = 0.1
    x = tf.reshape(x, [-1, dim_emit])
    ##
    params = []
    with tf.name_scope("variational_dist") as scope_parent:
        with tf.variable_scope("variational_dist_var") as v_scope_parent:
            layer, dim_out = build_nn(
                x,
                dim_input=dim_emit,
                n_steps=n_steps,
                hyparam_name="variational_internal_layers",
                name="vd",
                init_params_flag=init_params_flag,
                control_params=control_params,
            )
            sttype = control_params["config"]["state_type"]
            if sttype == "discrete" or sttype == "discrete_tr":
                with tf.variable_scope("vd_fc_logits") as scope:
                    layer_logit = layers.fc_layer(
                        "vd_fc_logits",
                        layer,
                        dim_out,
                        dim,
                        wd_w,
                        wd_bias,
                        activate=tf.tanh,
                        init_params_flag=init_params_flag,
                    )
                layer_logit = tf.reshape(layer_logit, [-1, n_steps, dim])
                layer_z = tf.nn.softmax(layer_logit)
                params.append(layer_z)
            elif sttype == "normal":
                with tf.variable_scope("vd_fc_mu") as scope:
                    layer_mu = layers.fc_layer(
                        "vd_fc_mu",
                        layer,
                        dim_out,
                        dim,
                        wd_w,
                        wd_bias,
                        activate=tf.tanh,
                        init_params_flag=init_params_flag,
                    )
                    layer_mu = tf.reshape(layer_mu, [-1, n_steps, dim])
                    params.append(layer_mu)
                with tf.variable_scope("vd_fc_cov") as scope:
                    pre_activate = layers.fc_layer(
                        "vd_fc_cov",
                        layer,
                        dim_out,
                        dim,
                        wd_w,
                        wd_bias,
                        activate=None,
                        init_params_flag=init_params_flag,
                    )
                    layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
                    max_var = control_params["config"]["normal_max_var"]
                    min_var = control_params["config"]["normal_min_var"]
                    layer_cov = tf.clip_by_value(layer_cov, min_var, max_var)
                    layer_cov = tf.reshape(layer_cov, [-1, n_steps, dim])
                    params.append(layer_cov)
            else:
                raise Exception("[Error] unknown state type")

    return params


# return parameters for p(x|z): list of tensors: batch_size x n_steps x dim
def computeEmission(z, n_steps, init_params_flag=True, control_params=None):
    """
	Return parameter for the conditional distribution p(x|z).
	Parmeters
	---------
	z: 
		bs x T x dim
		or (bs x T) x dim
	Returns
	-------
	output : list of tensors: batch_size x n_steps x dim
		parameters for p(x|z)
	"""

    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]
    wd_bias = None
    wd_w = 0.1
    hy_param = hy.get_hyperparameter()
    z = tf.reshape(z, [-1, dim])
    params = []
    with tf.name_scope("emission") as scope_parent:
        with tf.variable_scope("emission_var") as v_scope_parent:
            # z -> layer
            layer, dim_out = build_nn(
                z,
                dim_input=dim,
                n_steps=n_steps,
                hyparam_name="emission_internal_layers",
                name="em",
                init_params_flag=init_params_flag,
                control_params=control_params,
            )

            etype = control_params["config"]["emission_type"]
            if etype == "normal":
                # layer -> layer_mean
                with tf.variable_scope("em_fc_mean") as scope:
                    layer_mu = layers.fc_layer(
                        "emission/em_fc_mean",
                        layer,
                        dim_out,
                        dim_emit,
                        wd_w,
                        wd_bias,
                        activate=None,
                        init_params_flag=init_params_flag,
                    )
                layer_mu = tf.reshape(layer_mu, [-1, n_steps, dim_emit])
                params.append(layer_mu)
                # layer -> layer_cov
                with tf.variable_scope("em_fc_cov") as scope:
                    pre_activate = layers.fc_layer(
                        "emission/em_fc_cov",
                        layer,
                        dim_out,
                        dim_emit,
                        wd_w,
                        wd_bias,
                        activate=None,
                        init_params_flag=init_params_flag,
                    )
                    layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
                    max_var = control_params["config"]["normal_max_var"]
                    min_var = control_params["config"]["normal_min_var"]
                    layer_cov = tf.clip_by_value(layer_cov, min_var, max_var)
                layer_cov = tf.reshape(layer_cov, [-1, n_steps, dim_emit])
                params.append(layer_cov)
            elif etype == "binary":
                # layer -> sigmoid
                with tf.variable_scope("em_fc_out") as scope:
                    layer_logit = layers.fc_layer(
                        "emission/em_fc_out",
                        layer,
                        dim_out,
                        dim,
                        wd_w,
                        wd_bias,
                        activate=None,
                        init_params_flag=init_params_flag,
                    )
                    layer_out = tf.nn.sigmoid(layer_logit)
                layer_out = tf.reshape(layer_out, [-1, n_steps, dim_emit])
                params.append(layer_out)
            else:
                raise Exception("[Error] unknown emission type:" + etype)
    return params


# return parameters for p(l|z): list of tensors: batch_size x n_steps x dim
def computeLabel(z, n_steps, init_params_flag=True, control_params=None):
    """
	Return parameters for the conditonal distribution p(l|z).
	Parameters
	----------
		z: three dimension list # bs x T x dim or (bs x T) x dim
			z_{1:t}

	Returns
	-------
		params : ist of tensors: batch_size x n_steps x dim
			parameters for p(l|z)
	"""

    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_label = 2  # hy_param["dim_label"]
    wd_bias = None
    wd_w = 0.1
    hy_param = hy.get_hyperparameter()
    z = tf.reshape(z, [-1, dim])
    params = []
    with tf.name_scope("label") as scope_parent:
        with tf.variable_scope("label_var") as v_scope_parent:
            # z -> layer
            layer, dim_out = build_nn(
                z,
                dim_input=dim,
                n_steps=n_steps,
                hyparam_name="label_internal_layers",
                name="label",
                init_params_flag=init_params_flag,
                control_params=control_params,
            )

            ltype = control_params["config"]["label_type"]
            if ltype == "normal":
                # layer -> layer_mean
                with tf.variable_scope("label_fc_mean") as scope:
                    layer_mu = layers.fc_layer(
                        "label/label_fc_mean",
                        layer,
                        dim_out,
                        dim_label,
                        wd_w,
                        wd_bias,
                        activate=None,
                        init_params_flag=init_params_flag,
                    )
                layer_mu = tf.reshape(layer_mu, [-1, n_steps, dim_label])
                params.append(layer_mu)
                # layer -> layer_cov
                with tf.variable_scope("label_fc_cov") as scope:
                    pre_activate = layers.fc_layer(
                        "label/em_fc_cov",
                        layer,
                        dim_out,
                        dim_label,
                        wd_w,
                        wd_bias,
                        activate=None,
                        init_params_flag=init_params_flag,
                    )
                    layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
                    max_var = control_params["config"]["normal_max_var"]
                    min_var = control_params["config"]["normal_min_var"]
                    layer_cov = tf.clip_by_value(layer_cov, min_var, max_var)
                layer_cov = tf.reshape(layer_cov, [-1, n_steps, dim_label])
                params.append(layer_cov)
            elif ltype == "binary":
                # layer -> sigmoid
                with tf.variable_scope("label_fc_out") as scope:
                    layer_logit = layers.fc_layer(
                        "label/em_fc_out",
                        layer,
                        dim_out,
                        dim,
                        wd_w,
                        wd_bias,
                        activate=None,
                        init_params_flag=init_params_flag,
                    )
                    layer_out = tf.nn.sigmoid(layer_logit)
                layer_out = tf.reshape(layer_out, [-1, n_steps, dim_label])
                params.append(layer_out)
            elif ltype == "multinominal":
                # layer -> sigmoid
                with tf.variable_scope("label_fc_out") as scope:
                    layer_logit = layers.fc_layer(
                        "label/em_fc_out",
                        layer,
                        dim_out,
                        dim,
                        wd_w,
                        wd_bias,
                        activate=None,
                        init_params_flag=init_params_flag,
                    )
                    layer_out = layer_logit
                layer_out = tf.reshape(layer_out, [-1, n_steps, dim_label])
                params.append(layer_out)
            else:
                raise Exception("[Error] unknown label type:" + ltype)
    return params


def computeTransition(
    z, n_steps, init_state, init_params_flag=True, control_params=None
):
    """

	Parameters
	----------
		z :
		init_state : 
			the initional state.
	Returns
	-------
		out_param : 

	mu: bs x T x dim
	cov: bs x T x dim
	prior0: bs x 1 x dim
	"""
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    # add initial state and remove last state
    in_z = tf.reshape(z, [-1, n_steps, dim])
    in_z = tf.concat([init_state, in_z[:, :-1, :]], axis=1)

    out_param = computeTransitionDistWithNN(
        in_z, n_steps, init_params_flag, control_params=control_params
    )
    return out_param


# p(z_t|z_t+1)
def computeTransitionDistWithNN(
    in_points, n_steps, init_params_flag=True, control_params=None
):
    """
	Return parameters for the transition distribution p(z_t|z_{t-1})
	Parameters
	----------
		in_points : 
	Returns
	-------
		params : list
			layer_z : 
			layer_mu :
			layer_cov :

	"""
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]

    wd_bias = None
    wd_w = 0.1
    hy_param = hy.get_hyperparameter()
    z = tf.reshape(in_points, [-1, dim])
    params = []
    with tf.name_scope("transition") as scope_parent:
        with tf.variable_scope("transition_var") as v_scope_parent:
            layer = z
            dim_out = dim
            if hy_param["transition_internal_layers"]:
                # z -> layer
                layer, dim_out = build_nn(
                    layer,
                    dim_input=dim,
                    n_steps=n_steps,
                    hyparam_name="transition_internal_layers",
                    name="tr",
                    init_params_flag=init_params_flag,
                    control_params=control_params,
                )

            sttype = control_params["config"]["state_type"]
            if sttype == "discrete":
                # layer -> layer_mean
                with tf.variable_scope("tr_fc_logits") as scope:
                    layer_logit = layers.fc_layer(
                        "tr_fc_logits",
                        layer,
                        dim_out,
                        dim,
                        wd_w,
                        wd_bias,
                        activate=None,
                        init_params_flag=init_params_flag,
                    )
                    layer_z = tf.nn.softmax(layer_logit)
                    layer_z = tf.reshape(layer_z, [-1, n_steps, dim])
                    params.append(layer_z)
            elif sttype == "discrete_tr":
                # layer -> layer_mean
                with tf.variable_scope("tr_fc_logits") as scope:
                    layer_logit = layers.discrete_tr_layer(
                        "tr_fc_logits",
                        layer,
                        dim_out,
                        dim,
                        wd_w,
                        init_params_flag=init_params_flag,
                        beta=1.0,
                    )
                    layer_z = layer_logit
                    layer_z = tf.reshape(layer_z, [-1, n_steps, dim])
                    params.append(layer_z)

            elif sttype == "normal":
                with tf.variable_scope("vd_fc_mu") as scope:
                    layer_mu = layers.fc_layer(
                        "vd_fc_mu",
                        layer,
                        dim_out,
                        dim,
                        wd_w,
                        wd_bias,
                        activate=tf.tanh,
                        init_params_flag=init_params_flag,
                    )
                    layer_mu = tf.reshape(layer_mu, [-1, n_steps, dim])
                    params.append(layer_mu)
                with tf.variable_scope("vd_fc_cov") as scope:
                    pre_activate = layers.fc_layer(
                        "vd_fc_cov",
                        layer,
                        dim_out,
                        dim,
                        wd_w,
                        wd_bias,
                        activate=None,
                        init_params_flag=init_params_flag,
                    )
                    layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
                    max_var = control_params["config"]["normal_max_var"]
                    min_var = control_params["config"]["normal_min_var"]
                    layer_cov = tf.clip_by_value(layer_cov, min_var, max_var)
                    layer_cov = tf.reshape(layer_cov, [-1, n_steps, dim])
                    params.append(layer_cov)
            else:
                raise Exception("[Error] unknown emission type:" + sttype)
    return params


def computeTransitionFunc(
    in_points, n_steps, init_params_flag=True, control_params=None
):
    """
	# x_{t+1} = f (x_t)
	Parameters
	----------
		in_points: points x dim

	Returns
	-------
		parameters for p(z_t|z_t-1): list of tensors: batch_size x n_steps x dim
	"""
    hy_param = hy.get_hyperparameter()
    if hy_param["potential_grad_transition_enabled"]:
        return computeTransitionFuncFromPotential(
            in_points, n_steps, init_params_flag, control_params
        )
    else:
        return computeTransitionFuncFromNN(
            in_points, n_steps, init_params_flag, control_params
        )


def computeTransitionFuncFromPotential(
    in_points, n_steps, init_params_flag=True, control_params=None
):
    """
	Parmeters
	---------
		in_points: points x dim
	
	Returns
	-------
		laeyer_mean : 
	"""
    with tf.name_scope("transition") as scope_parent:
        # z  : (bs x T) x dim
        # pot: (bs x T)
        pot = computePotential(
            in_points,
            n_steps,
            init_params_flag=init_params_flag,
            control_params=control_params,
        )
        sum_pot = tf.reduce_sum(pot)
        g_z = tf.gradients(sum_pot, [in_points])
        # print(g_z)
        layer_mean = in_points + g_z
    return layer_mean


"""
# x_t+1 = f (x_t)
in_points: points x dim
return:
points x dim
"""


def computeTransitionFuncFromNN(
    in_points, n_steps, init_params_flag=True, control_params=None
):
    """
	Parameters
	----------
		in_points :

	Returns
	-------
		layer_z :
	"""
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]

    wd_bias = None
    wd_w = 0.1
    hy_param = hy.get_hyperparameter()
    layer = tf.reshape(in_points, [-1, dim])
    with tf.name_scope("transition") as scope_parent:
        with tf.variable_scope("transition_var") as v_scope_parent:
            if hy_param["transition_internal_layers"]:
                # z -> layer
                layer, dim_out = build_nn(
                    layer,
                    dim_input=dim,
                    n_steps=n_steps,
                    hyparam_name="transition_internal_layers",
                    name="tr",
                    init_params_flag=init_params_flag,
                    control_params=control_params,
                )

            # layer -> layer_mean
            with tf.variable_scope("tr_fc_out") as scope:
                layer_logit = layers.fc_layer(
                    "tr_fc_out",
                    layer,
                    dim_out,
                    dim,
                    wd_w,
                    wd_bias,
                    activate=None,
                    init_params_flag=init_params_flag,
                )
            sttype = control_params["config"]["state_type"]
            if sttype == "discrete" or sttype == "discrete_tr":
                layer_z = tf.nn.softmax(layer_logit)
            elif sttype == "normal":
                layer_z = layer_logit
            else:
                raise Exception("[Error] unknown state type")
    return layer_z


def sampleTransitionFromDist(
    z_param, n_steps, init_state, init_params_flag=True, control_params=None
):
    """
		Parameters
		----------
		z_param :

		Returns
		-------
		q_zz :

		z_s  :
			sampled points from Normal distribution whose parameters are q_zz.

	"""
    sttype = control_params["config"]["state_type"]
    if sttype == "normal":
        eps = control_params["placeholders"]["tr_eps"]
        q_zz = computeTransitionUKF(
            z_param[0],
            z_param[1],
            n_steps,
            mean_prior0=init_state[0],
            cov_prior0=init_state[1],
            init_params_flag=init_params_flag,
            control_params=control_params,
        )
        z_s = sample_normal(q_zz, eps)
    else:
        raise Exception(
            "[Error] not supported dynamics_type=function & state_type=%s" % (sttype,)
        )
    return z_s, q_zz


def sampleTransition(
    z, n_steps, init_state, init_params_flag=True, control_params=None
):
    """
		Parameters
		----------
			z :

		Returns
		-------
			z_s :

			q_zz:
	"""
    sttype = control_params["config"]["state_type"]
    if sttype == "discrete" or sttype == "discrete_tr":
        q_zz = computeTransition(
            z,
            n_steps,
            init_state,
            init_params_flag=init_params_flag,
            control_params=control_params,
        )
        # z_s=q_zz[0]
        dist = tf.contrib.distributions.OneHotCategorical(probs=q_zz[0])
        z_s = tf.cast(dist.sample(), tf.float32)
    elif sttype == "normal":
        q_zz = computeTransition(
            z,
            n_steps,
            init_state,
            init_params_flag=init_params_flag,
            control_params=control_params,
        )
        eps = control_params["placeholders"]["tr_eps"]
        z_s = sample_normal(q_zz, eps)
    else:
        raise Exception("[Error] unknown state type")
    return z_s, q_zz


def p_filter(
    x,
    z,
    m,
    step,
    n_steps,
    epsilon,
    sample_size,
    proposal_sample_size,
    batch_size,
    control_params,
):
    """
	Parameters
	----------
		x :
		z :
		m :
	"""
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]

    resample_size = sample_size
    ptype = control_params["config"]["pfilter_type"]
    dytype = control_params["config"]["dynamics_type"]
    sttype = control_params["config"]["state_type"]
    print(z.shape)
    if ptype == "trained_dynamics":
        if dytype == "function":
            #  z: (sample_size x batch_size) x n_step x  dim
            mu_trans = computeTransitionFunc(z, n_steps, control_params=control_params)
            #  m: sample_size x batch_size x dim
            m = tf.reshape(mu_trans, [sample_size, -1, n_step, dim])
            m = m[:, :, step, :]
            d = m - tf.reduce_mean(m, axis=0)
            cov = tf.reduce_mean(d ** 2, axis=0)
            cov_trans = tf.tile(cov, [sample_size, 1])
            #  mu_trans : (sample_size x batch_size) x dim
            #  cov_trans: (sample_size x batch_size) x dim
            mu_trans = tf.reshape(mu_trans, [-1, dim])
            cov_trans = tf.reshape(cov_trans, [-1, dim]) + 0.01
            #
            proposal_dist = tf.contrib.distributions.Normal(
                mu_trans[:, :], cov_trans[:, :]
            )
            particles = proposal_dist.sample(proposal_sample_size)
        elif dytype == "distribution":
            out_params = computeTransitionDistWithNN(
                z, n_steps, init_params_flag=True, control_params=control_params
            )
            if sttype == "discrete":
                logit = out_params[0][:, step, :]
                logpi = tf.log(logit + 1.0e-10)
                proposal_dist = tf.contrib.distributions.OneHotCategorical(logits=logpi)
                particles = tf.cast(
                    proposal_dist.sample(proposal_sample_size), tf.float32
                )
            elif sttype == "discrete_tr":
                probs = out_params[0][:, step, :]
                proposal_dist = tf.contrib.distributions.OneHotCategorical(probs=probs)
                particles = tf.cast(
                    proposal_dist.sample(proposal_sample_size), tf.float32
                )
            elif sttype == "normal":
                mu_trans = out_params[0][:, step, :]
                cov_trans = out_params[1][:, step, :] + 0.01
                proposal_dist = tf.contrib.distributions.Normal(mu_trans, cov_trans)
                particles = proposal_dist.sample(proposal_sample_size)
            else:
                raise Exception("[Error] unknown state type")
        else:
            raise Exception("[Error] unknown dynamics type")
    elif ptype == "zero_dynamics":
        var = control_params["config"]["zero_dynamics_var"]
        proposal_dist = tf.contrib.distributions.Normal(z, var)
        particles = proposal_dist.sample(proposal_sample_size)
        # 1000, ?, n_steps, dim
        particles = particles[:, :, step, :]
    else:
        raise Exception("[Error] unknown pfilter type")

    #  particles: proposal_sample_size x (sample_size x batch_size) x dim
    # particles_d=particles-mu_trans[:,:]
    # particles_w=particles_d**2/cov_trans[:,:]
    #  particles: (proposal_sample_size x sample_size x batch_size) x dim
    print("@@@@", particles.shape)
    particles = tf.reshape(particles, [-1, dim])
    print("@@@@", particles.shape)
    obs_params = computeEmission(particles, 1, control_params=control_params)
    print("@@@@", batch_size)
    print("@@@@", proposal_sample_size, sample_size)
    #  mu: (proposal_sample_size x sample_size)  x batch_size x emit_dim
    #  cov: (proposal_sample_size x sample_size) x batch_size x emit_dim
    mu = obs_params[0]
    cov = obs_params[1]
    mu = tf.reshape(mu, [-1, batch_size, dim_emit])
    cov = tf.reshape(cov, [-1, batch_size, dim_emit])
    #  x: batch_size x emit_dim
    d = (mu - x[:, :]) * m
    w = -tf.reduce_sum(d ** 2 / cov, axis=2)
    # w:(proposal_sample_size x sample_size) x batch__size
    #  probs=w/tf.reduce_sum(w,axis=0)

    resample_dist = tf.contrib.distributions.Categorical(logits=tf.transpose(w))
    # ids: resample x batch_size
    #  particles: (proposal_sample_size x sample_size) x batch_size x dim
    particle_ids = resample_dist.sample([resample_size])
    particles = tf.reshape(
        particles, [proposal_sample_size * sample_size, batch_size, dim]
    )
    #
    dummy = np.zeros((resample_size, batch_size, 1), dtype=np.int32)
    particle_ids = tf.reshape(particle_ids, [resample_size, batch_size, 1])
    for i in range(batch_size):
        dummy[:, i, 0] = i
    temp = tf.constant(dummy)
    particle_ids = tf.concat([particle_ids, temp], 2)
    # particles: (resample) x b x dim
    out = tf.gather_nd(particles, particle_ids)
    outputs = {"sampled_pred_params": [mu, cov], "sampled_z": out}
    return outputs


def get_init_state(batch_size, dim):
    init_s = np.zeros((1, 1, dim), dtype=np.float32)
    init_s[:, :, 0] = 1
    init_state = tf.tile(tf.constant(init_s, dtype=np.float32), (batch_size, 1, 1))
    return init_state


def get_init_dist(batch_size, dim):
    """
	Get inital distribution whose the mean parameter is 0 (vector) and the variance parameter is 1 (vector).
	Parameters
	----------
		batch_size : 
		dim :
	Returns
	-------
		[init_mu,init_var] : list
			init_mu :
				the mean parameter of the distribution
			init_var :
				the variance parameter of the distribution
	"""
    init_m = np.zeros((1, 1, dim), dtype=np.float32)
    init_mu = tf.tile(tf.constant(init_m, dtype=np.float32), (batch_size, 1, 1))
    init_v = np.ones((1, 1, dim), dtype=np.float32)
    init_var = tf.tile(tf.constant(init_v, dtype=np.float32), (batch_size, 1, 1))
    return [init_mu, init_var]


def inference(n_steps, control_params):
    """
	Returns
	-------
	inference results by sampling or function.
	"""
    dytype = control_params["config"]["dynamics_type"]
    if dytype == "distribution":
        return inference_by_sample(n_steps, control_params)
    elif dytype == "function":
        return inference_by_dist(n_steps, control_params)
    else:
        raise Exception("[Error] unknown dynamics type")


def inference_by_dist(n_steps, control_params):
    """
	Returns
	-------
		outputs :
			indference results
	"""
    # get input data
    placeholders = control_params["placeholders"]
    x = placeholders["x"]
    pot_points = placeholders["potential_points"]
    # get parameters
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]
    bs = tf.shape(x)[0]
    with tf.name_scope("inference") as scope_parent:
        # z_q: (bs x T) x dim
        z_s, z_params = sampleVariationalDist(
            x, n_steps, init_params_flag=True, control_params=control_params
        )
        init_state = get_init_dist(bs, dim)
        z_pred_s, z_pred_params = sampleTransitionFromDist(
            z_params,
            n_steps,
            init_state,
            init_params_flag=True,
            control_params=control_params,
        )
        pot_loss = None
        # compute emission
        obs_params = computeEmission(
            z_s, n_steps, init_params_flag=True, control_params=control_params
        )
        obs_pred_params = computeEmission(
            z_pred_s, n_steps, init_params_flag=False, control_params=control_params
        )
    outputs = {
        "z_s": z_s,
        "z_params": z_params,
        "z_pred_s": z_pred_s,
        "z_pred_params": z_pred_params,
        "obs_params": obs_params,
        "obs_pred_params": obs_pred_params,
        "potential_loss": pot_loss,
    }
    return outputs


def inference_by_sample(n_steps, control_params):
    """
	Returns
	-------
	output :
		indference results
	"""
    # get input data
    placeholders = control_params["placeholders"]
    x = placeholders["x"]
    pot_points = placeholders["potential_points"]
    # get parameters
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]
    bs = tf.shape(x)[0]
    with tf.name_scope("inference") as scope_parent:
        # z_q: (bs x T) x dim
        z_s, z_params = sampleVariationalDist(
            x, n_steps, init_params_flag=True, control_params=control_params
        )
        init_state = get_init_state(bs, dim)
        z_pred_s, z_pred_params = sampleTransition(
            z_s,
            n_steps,
            init_state,
            init_params_flag=True,
            control_params=control_params,
        )
        pot_loss = None
        # compute emission
        obs_params = computeEmission(
            z_s, n_steps, init_params_flag=True, control_params=control_params
        )
        obs_pred_params = computeEmission(
            z_pred_s, n_steps, init_params_flag=False, control_params=control_params
        )
    outputs = {
        "z_s": z_s,
        "z_params": z_params,
        "z_pred_s": z_pred_s,
        "z_pred_params": z_pred_params,
        "obs_params": obs_params,
        "obs_pred_params": obs_pred_params,
        "potential_loss": pot_loss,
    }
    return outputs


def inference_label(outputs, n_steps, control_params):
    """
	Parameters
	----------
		outputs :
	Returns
	-------
		outputs :
	"""
    assert "z_s" in outputs, "outputs does not contain z_s"
    z_s = outputs["z_s"]
    params = computeLabel(
        z_s, n_steps, init_params_flag=True, control_params=control_params
    )
    outputs["label_params"] = params
    return outputs


def computeNegCLL(x, params, mask, control_params):
    """
	Prameters
	---------
		x : 
		params : list
		mask :
	Retunrs
	-------
		negCLL
	"""
    mu_p = params[0]
    cov_p = params[1]

    negCLL = tf.log(2 * np.pi) + tf.log(cov_p) + (x - mu_p) ** 2 / cov_p
    negCLL = negCLL * 0.5
    negCLL = negCLL * mask
    negCLL = tf.reduce_sum(negCLL, axis=2)
    negCLL = tf.reduce_sum(negCLL, axis=1)
    return negCLL


def kl_normal(mu1, var1, mu2, var2):
    """
	KL(N() ||  N())
	"""
    return (
        tf.log(var2) / 2.0
        - tf.log(var1) / 2.0
        + var1 / (2.0 * var2)
        + (mu1 - mu2) ** 2 / (2.0 * var2)
        - 1 / 2
    )


def computeTemporalKL(x, outputs, length, control_params):
    """
	Parameters
	----------
		x :
		outputs :
		length :
	Retunrs
	-------
		kl_t :
	"""
    sttype = control_params["config"]["state_type"]
    if sttype == "discrete" or sttype == "discrete_tr":
        mu_p = outputs["z_pred_params"][0]
        mu_q = outputs["z_params"][0]
        eps = 1.0e-10
        kl_t = mu_q * (tf.log(mu_q + eps) - tf.log(mu_p + eps))
    elif sttype == "normal":
        cov_p = outputs["z_pred_params"][1]
        mu_p = outputs["z_pred_params"][0]
        cov_q = outputs["z_params"][1]
        mu_q = outputs["z_params"][0]
        kl_t = tf.log(cov_p) - tf.log(cov_q) - 1 + (cov_q + (mu_p - mu_q) ** 2) / cov_p
    else:
        raise Exception("[Error] unknown state type")

    # masked_kl=tf.reduce_sum(kl_t,axis=2)*mask
    mask = tf.sequence_mask(length, maxlen=kl_t.shape[1], dtype=tf.float32)
    kl_t = tf.reduce_sum(kl_t, axis=2)
    kl_t = kl_t * mask

    kl_t = tf.reduce_sum(kl_t, axis=1)
    return kl_t


def loss(outputs, alpha=1, control_params=None):
    """
	Parameters
	----------
		outputs :
		alpha :
	Returns
	-------
		Loss : tensor of type float.
	"""
    # get input data
    placeholders = control_params["placeholders"]
    x = placeholders["x"]
    mask = placeholders["m"]
    length = placeholders["s"]
    # loss
    negCLL = computeNegCLL(x, outputs["obs_params"], mask, control_params)
    temporalKL = computeTemporalKL(x, outputs, length, control_params)
    cost_pot = tf.constant(0.0, dtype=np.float32)

    costs_name = ["recons", "temporal"]
    costs = [tf.reduce_mean(negCLL), tf.reduce_mean(temporalKL)]
    errors_name = []
    errors = []
    updates = []
    metrics_name = []
    metrics = []
    if outputs["potential_loss"] is not None:
        pot = outputs["potential_loss"]
        # sum_pot=tf.reduce_sum(pot*mask,axis=1)
        sum_pot = tf.reduce_sum(pot, axis=1)
        cost_pot = tf.reduce_mean(pot)
        ##
        costs_name.append("potential")
        costs.append(cost_pot)

    cost_label = tf.constant(0.0, dtype=np.float32)
    if "label_params" in outputs and outputs["label_params"] is not None:
        ltype = control_params["config"]["label_type"]
        assert ltype == "multinominal", "not supported label type"
        label = placeholders["l"]
        logit = outputs["label_params"][0]
        cost_label = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label, logits=logit
        )

        label_accuracy, update_op = tf.metrics.accuracy(label, tf.argmax(logit, axis=2))
        cost_label = tf.reduce_sum(cost_label, axis=1)
        ##
        costs_name.append("label")
        costs.append(tf.reduce_mean(cost_label))
        ##
        metrics_name.append("label_acc")
        metrics.append(label_accuracy)
        updates.append(update_op)

    if "obs_params" in outputs:
        diff = tf.reduce_mean((x - outputs["obs_params"][0]) ** 2)
        ##
        errors_name.append("recons_mse")
        errors.append(diff)

    mean_cost = tf.reduce_mean(
        (negCLL + alpha * temporalKL + alpha * 1.0 * cost_pot) + cost_label,
        name="train_cost",
    )
    tf.add_to_collection("losses", mean_cost)
    total_cost = tf.add_n(tf.get_collection("losses"), name="total_loss")

    return {
        "cost": total_cost,
        "mean_cost": mean_cost,
        "all_costs": costs,
        "all_costs_name": costs_name,
        "errors": errors,
        "errors_name": errors_name,
        "metrics": metrics,
        "metrics_name": metrics_name,
        "updates": updates,
    }


def _add_loss_summaries(total_loss):
    """
	Generates moving average for all losses and associated summaries for
	visualizing the performance of the network.

	Args:
		total_loss: Total loss from loss().
	Returns:
		loss/_averages_op: op for generating moving averages of losses.
	"""
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name="avg")
    losses = tf.get_collection("losses")
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + " (raw)", l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def computeTransitionUKF(
    mu,
    cov,
    n_steps,
    mean_prior0=None,
    cov_prior0=None,
    init_params_flag=True,
    control_params=None,
):
    """
	Uncented Kalman Filter
	Parameters
	----------
		mu: 
			bs x T x dim or (bs x T) x dim
		cov: 
			bs x T x dim or (bs x T) x dim
		mean_prior0 :
			 bs x 1 x dim
		cov_prior0 : 
	Returns
	-------
		output_mu: 
			bs x T x dim
		output_cov: 
			bs x T x dim
		
	###
	if prior0==None
		z[1:T+1]
	else
		z[0:T]
	"""
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    # in_mu=tf.reshape(mu,[1,-1,dim])
    in_mu = tf.reshape(mu, [-1, dim])
    in_sigma = tf.sqrt(tf.reshape(cov, [-1, dim]))
    # in_points=tf.tile(in_mu,[dim*2+1,1,1])
    in_points = [in_mu]
    for i in range(dim):
        in_points.append(in_mu + in_sigma)
        in_points.append(in_mu - in_sigma)
    in_points = tf.stack(in_points)

    out_points = computeTransitionFunc(
        in_points,
        n_steps,
        init_params_flag=init_params_flag,
        control_params=control_params,
    )
    # N x bs x T x dim
    layer_mean = tf.reshape(out_points, [dim * 2 + 1, -1, n_steps, dim])
    temp_sigma = layer_mean - layer_mean[0, :, :, :]
    layer_cov = tf.reduce_sum(temp_sigma ** 2, axis=0)

    if mean_prior0 is not None and cov_prior0 is not None:
        output_mu = tf.concat([mean_prior0, layer_mean[0, :, :-1, :]], axis=1)
        output_cov = tf.concat([cov_prior0, layer_cov[:, :-1, :]], axis=1)
        return output_mu, output_cov
    else:
        return layer_mean[0, :, :, :], layer_cov


def computePotentialLoss(mu_q, cov_q, pot_points, n_steps, control_params=None):
    """
	Parameters
	----------
		mu_q :
			bs x T x dim
		cov_q :

		pot_points :

	Returns
	-------
		pot_loss :

	"""
    pot_loss = None
    if hy_param["potential_enabled"]:
        if hy_param["potential_grad_transition_enabled"] == False:
            use_data_points = False
            if pot_points is None:
                use_data_points = True
            if use_data_points:
                ## compute V(x(t+1))-V(x(t)) < 0 for stability
                mu_trans_1, cov_trans_1 = computeTransitionUKF(
                    mu_q,
                    cov_q,
                    n_steps,
                    None,
                    None,
                    init_params_flag=False,
                    control_params=control_params,
                )
                # mu_q: bs x T x dim
                # mu_trans_1: bs x T x dim
                params_pot = None
                pot0 = computePotential(mu_q, n_steps, control_params=control_params)
                pot1 = computePotential(
                    mu_trans_1,
                    n_steps,
                    init_params_flag=False,
                    control_params=control_params,
                )
                # pot: bs x T
                c = 0.1
                pot = tf.nn.relu(pot1 - pot0 + c)
                pot_loss = tf.reshape(pot, [-1, n_steps])
            else:
                mu_trans_1 = computeTransitionFunc(
                    pot_points, 1, init_params_flag=False, control_params=control_params
                )
                params_pot = None
                pot0 = computePotential(pot_points, 1, control_params=control_params)
                pot1 = computePotential(
                    mu_trans_1, 1, init_params_flag=False, control_params=control_params
                )
                # pot: bs x T
                c = 0.1
                pot = tf.nn.relu(pot1 - pot0 + c)
                pot_loss = tf.reshape(pot, [-1, 1])
    return pot_loss


"""
z: (bs x T) x dim
  or bs x T x dim
return:	
pot: (bs x T)
"""


def computePotential(z_input, n_steps, init_params_flag=True, control_params=None):
    """
		Parameters
		----------
			z_input :

		Returns
		-------

	"""
    hy_param = hy.get_hyperparameter()
    if hy_param["potential_nn_enabled"]:
        return computePotentialFromNN(
            z_input, n_steps, init_params_flag, control_params
        )
    else:
        return computePotentialWithBinaryPot(
            z_input, n_steps, init_params_flag, control_params
        )


"""
z: (bs x T) x dim
  or bs x T x dim
return:
pot: (bs x T)
"""


def computePotentialWithBinaryPot(
    z_input, n_steps, dim, init_params_flag=True, control_params=None
):
    """
	Parameters
	----------
		z_input :

	Returns
	-------
		pot :

	"""
    pot_pole = []

    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    z = tf.reshape(z_input, [-1, dim])
    # z=tf.reshape(z_input,[900,dim])
    for d in range(dim):
        z1 = np.zeros((dim,), dtype=np.float32)
        z2 = np.zeros((dim,), dtype=np.float32)
        z1[d] = 0.5
        z2[d] = -0.5

        z1 = z - tf.constant(z1, dtype=np.float32)
        z2 = z - tf.constant(z2, dtype=np.float32)
        p1 = tf.reduce_sum(z1 ** 2, axis=1)
        p2 = tf.reduce_sum(z2 ** 2, axis=1)
        pot_pole.append(p1)
        pot_pole.append(p2)
    pot_pole = tf.stack(pot_pole)
    # pot_pole: (2xdim) x (bs x T)
    pot = tf.reduce_min(pot_pole, axis=0)
    return pot


"""
z: (bs x T) x dim
  or bs x T x dim
return:
pot: (bs x T)
"""


def computePotentialFromNN(
    z_input, n_steps, init_params_flag=True, control_params=None
):
    """
	Compute the value of the potential V(z_t).
	Parameters
	----------
		z_input :

	Returns
	-------
		layer_mean : 

	"""
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]

    wd_bias = None
    wd_w = 0.1
    hy_param = hy.get_hyperparameter()
    z = tf.reshape(z_input, [-1, dim])
    with tf.name_scope("potential") as scope_parent:
        with tf.variable_scope("potential_var") as v_scope_parent:
            if hy_param["potential_internal_layers"]:
                # z -> layer
                layer, dim_out = build_nn(
                    z,
                    dim_input=dim,
                    n_steps=n_steps,
                    hyparam_name="potential_internal_layers",
                    name="pot",
                    init_params_flag=init_params_flag,
                    control_params=control_params,
                )
                # layer -> layer_mean
                with tf.variable_scope("pot_fc_mean") as scope:
                    layer_mean = layers.fc_layer(
                        "pot_fc_mean",
                        layer,
                        dim_out,
                        1,
                        wd_w,
                        wd_bias,
                        activate=tf.sigmoid,
                        init_params_flag=init_params_flag,
                    )
            else:
                layer = z
                layer_mean = layers.fc_layer(
                    "pot_fc_mean",
                    layer,
                    dim,
                    1,
                    wd_w,
                    wd_bias,
                    activate=tf.sigmoid,
                    init_params_flag=init_params_flag,
                )
    layer_mean = tf.reshape(layer_mean, [-1])
    return layer_mean


#  x0: batch_size x sample_size x emit_dim
#  x: batch_size x n_steps x emit_dim
def fivo(
    x,
    x0,
    epsilon,
    n_steps,
    sample_size,
    proposal_sample_size,
    batch_size,
    control_params,
):
    """

	Parameters
	----------
		x0: 
			batch_size x sample_size x emit_dim
		x: 
			batch_size x n_steps x emit_dim
		sample_size :
		prposal_sample_size :

	Returns
	-------
		output : dict

	"""
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]

    resample_size = sample_size
    ptype = control_params["config"]["pfilter_type"]
    dytype = control_params["config"]["dynamics_type"]
    sttype = control_params["config"]["state_type"]
    zs = []
    x1 = x0
    obs_list = [[], []]
    init_params_flag = True
    for s in range(n_steps):
        if dytype == "distribution":
            # return parameters for q(z): list of tensors: batch_size x n_steps x dim
            if s == 0:
                qz = computeVariationalDist(x1, 1, init_params_flag, control_params)
            else:
                x1 = x[:, :s, :]
                qz = computeVariationalDist(x1, s, init_params_flag, control_params)
                print(qz[0].shape)
                qz[0] = qz[0][:, s - 1, :]
                qz[1] = qz[1][:, s - 1, :]
            qs = sampleState(qz, control_params, (sample_size))
            particles = tf.cast(qs, tf.float32)
            print(">>", particles.shape)
        else:
            raise Exception("[Error] unsupported dynamics type")

        #  particles: proposal_sample_size x (sample_size x batch_size) x dim
        #  particles: (proposal_sample_size x sample_size x batch_size) x dim
        particles = tf.reshape(particles, [-1, dim])
        obs_params = computeEmission(
            particles,
            1,
            init_params_flag=init_params_flag,
            control_params=control_params,
        )
        #  mu: (proposal_sample_size x sample_size)  x batch_size x emit_dim
        #  cov: (proposal_sample_size x sample_size) x batch_size x emit_dim
        mu = obs_params[0]
        cov = obs_params[1]
        mu = tf.reshape(mu, [-1, batch_size, dim_emit])
        cov = tf.clip_by_value(tf.reshape(cov, [-1, batch_size, dim_emit]), 1.0e-10, 2)
        #  x: batch_size x n_steps x emit_dim
        d = mu - x[:, s, :]
        w = -tf.reduce_sum(d ** 2 / cov, axis=2)
        # w:(proposal_sample_size x sample_size) x batch__size
        #  probs=w/tf.reduce_sum(w,axis=0)

        resample_dist = tf.contrib.distributions.Categorical(logits=tf.transpose(w))
        # ids: resample x batch_size
        #  particles: (proposal_sample_size x sample_size) x batch_size x dim
        particle_ids = resample_dist.sample([resample_size])
        particles = tf.reshape(particles, [-1, batch_size, dim])
        #
        dummy = np.zeros((resample_size, batch_size, 1), dtype=np.int32)
        particle_ids = tf.reshape(particle_ids, [resample_size, batch_size, 1])
        for i in range(batch_size):
            dummy[:, i, 0] = i
        temp = tf.constant(dummy)
        particle_ids = tf.concat([particle_ids, temp], 2)
        # out_z: (resample) x batch_size x dim
        # out_mu: (resample)  x batch_size x emit_dim
        out_z = tf.gather_nd(particles, particle_ids)
        out_mu = tf.gather_nd(mu, particle_ids)
        print("mu", mu.shape)
        print("out_mu", out_mu.shape)
        out_cov = tf.gather_nd(cov, particle_ids)
        obs_list[0].append(out_mu)
        obs_list[1].append(out_cov)
        print("out_z", out_z.shape)
        zs.append(out_z)
        ##
        init_params_flag = False
        z = out_z
    # mu: T x (resample)  x batch_size x emit_dim
    # mu_p: (resample)  x batch_size x T x emit_dim
    # zs: (resample)  x batch_size x T x dim
    mu_p = tf.transpose(tf.stack(obs_list[0]), perm=[1, 2, 0, 3])
    cov_p = tf.transpose(tf.stack(obs_list[1]), perm=[1, 2, 0, 3])
    # zz=tf.transpose(tf.stack(zs),perm=[1,2,0,3])
    #  x: batch_size x n_steps x emit_dim
    negCLL = tf.log(2 * np.pi) + tf.log(cov_p) + (x - mu_p) ** 2 / cov_p
    negCLL = negCLL * 0.5
    negCLL = tf.reduce_mean(negCLL, axis=0)  # sample
    negCLL = tf.reduce_sum(negCLL, axis=2)  # dim
    negCLL = tf.reduce_sum(negCLL, axis=1)  # T

    output_cost = {
        "cost": tf.reduce_mean(negCLL),
        "error": tf.reduce_mean(negCLL),
        "all_costs": tf.reduce_mean(negCLL),
    }
    # loss
    # outputs={
    # 	"sampled_pred_params":obs_list,
    # 	"sampled_z":zs}
    return output_cost


#  z0: sample_size x dim
#  x: batch_size x n_steps x emit_dim
def fivo2(
    x,
    z0,
    epsilon,
    n_steps,
    sample_size,
    proposal_sample_size,
    batch_size,
    control_params,
):
    hy_param = hy.get_hyperparameter()
    dim = hy_param["dim"]
    dim_emit = hy_param["dim_emit"]

    resample_size = sample_size
    ptype = control_params["config"]["pfilter_type"]
    dytype = control_params["config"]["dynamics_type"]
    sttype = control_params["config"]["state_type"]
    z = z0
    zs = []
    obs_list = [[], []]
    print(z.shape)
    init_params_flag = True
    for s in range(n_steps):
        if ptype == "trained_dynamics":
            if dytype == "function":
                #  x: (batch_size x T) x dim_emit
                #  z: (sample_size x batch_size) x dim
                mu_trans = computeTransitionFunc(
                    z,
                    1,
                    init_params_flag=init_params_flag,
                    control_params=control_params,
                )
                #  m: sample_size x batch_size x dim
                m = tf.reshape(mu_trans, [sample_size, -1, dim])
                d = m - tf.reduce_mean(m, axis=0)
                cov = tf.reduce_mean(d ** 2, axis=0)
                cov_trans = tf.tile(cov, [sample_size, 1])
                #  mu_trans : (sample_size x batch_size) x dim
                #  cov_trans: (sample_size x batch_size) x dim
                mu_trans = tf.reshape(mu_trans, [-1, dim])
                cov_trans = tf.reshape(cov_trans, [-1, dim]) + 0.01
                #
                proposal_dist = tf.contrib.distributions.Normal(
                    mu_trans[:, :], cov_trans[:, :]
                )
                particles = proposal_dist.sample(proposal_sample_size)
            elif dytype == "distribution":
                out_params = computeTransitionDistWithNN(
                    z,
                    1,
                    init_params_flag=init_params_flag,
                    control_params=control_params,
                )
                if sttype == "discrete":
                    logpi = tf.log(out_params[0] + 1.0e-10)
                    proposal_dist = tf.contrib.distributions.OneHotCategorical(
                        logits=logpi
                    )
                    particles = tf.cast(
                        proposal_dist.sample(proposal_sample_size), tf.float32
                    )
                elif sttype == "discrete_tr":
                    proposal_dist = tf.contrib.distributions.OneHotCategorical(
                        probs=out_params[0]
                    )
                    particles = tf.cast(
                        proposal_dist.sample(proposal_sample_size), tf.float32
                    )
                elif sttype == "normal":
                    mu_trans = out_params[0]
                    cov_trans = out_params[1] + 0.01
                    proposal_dist = tf.contrib.distributions.Normal(
                        mu_trans[:, 0, :], cov_trans[:, 0, :]
                    )
                    particles = proposal_dist.sample(proposal_sample_size)
                else:
                    raise Exception("[Error] unknown state type")
            else:
                raise Exception("[Error] unknown dynamics type")
        elif ptype == "zero_dynamics":
            var = control_params["config"]["zero_dynamics_var"]
            proposal_dist = tf.contrib.distributions.Normal(z, var)
            particles = proposal_dist.sample(proposal_sample_size)
        else:
            raise Exception("[Error] unknown pfilter type")

        #  particles: proposal_sample_size x (sample_size x batch_size) x dim
        #  particles: (proposal_sample_size x sample_size x batch_size) x dim
        particles = tf.reshape(particles, [-1, dim])
        obs_params = computeEmission(
            particles,
            1,
            init_params_flag=init_params_flag,
            control_params=control_params,
        )
        #  mu: (proposal_sample_size x sample_size)  x batch_size x emit_dim
        #  cov: (proposal_sample_size x sample_size) x batch_size x emit_dim
        mu = obs_params[0]
        cov = obs_params[1]
        mu = tf.reshape(mu, [-1, batch_size, dim_emit])
        cov = tf.clip_by_value(tf.reshape(cov, [-1, batch_size, dim_emit]), 1.0e-10, 2)
        #  x: batch_size x n_steps x emit_dim
        d = mu - x[:, s, :]
        w = -tf.reduce_sum(d ** 2 / cov, axis=2)
        # w:(proposal_sample_size x sample_size) x batch__size
        #  probs=w/tf.reduce_sum(w,axis=0)

        resample_dist = tf.contrib.distributions.Categorical(logits=tf.transpose(w))
        # ids: resample x batch_size
        #  particles: (proposal_sample_size x sample_size) x batch_size x dim
        particle_ids = resample_dist.sample([resample_size])
        particles = tf.reshape(particles, [-1, batch_size, dim])
        #
        dummy = np.zeros((resample_size, batch_size, 1), dtype=np.int32)
        particle_ids = tf.reshape(particle_ids, [resample_size, batch_size, 1])
        for i in range(batch_size):
            dummy[:, i, 0] = i
        temp = tf.constant(dummy)
        particle_ids = tf.concat([particle_ids, temp], 2)
        # out_z: (resample) x batch_size x dim
        # out_mu: (resample)  x batch_size x emit_dim
        out_z = tf.gather_nd(particles, particle_ids)
        out_mu = tf.gather_nd(mu, particle_ids)
        print("mu", mu.shape)
        print("out_mu", out_mu.shape)
        out_cov = tf.gather_nd(cov, particle_ids)
        obs_list[0].append(out_mu)
        obs_list[1].append(out_cov)
        print("out_z", out_z.shape)
        zs.append(out_z)
        ##
        init_params_flag = False
        z = out_z
    # mu: T x (resample)  x batch_size x emit_dim
    # mu_p: (resample)  x batch_size x T x emit_dim
    # zs: (resample)  x batch_size x T x dim
    mu_p = tf.transpose(tf.stack(obs_list[0]), perm=[1, 2, 0, 3])
    cov_p = tf.transpose(tf.stack(obs_list[1]), perm=[1, 2, 0, 3])
    # zz=tf.transpose(tf.stack(zs),perm=[1,2,0,3])
    #  x: batch_size x n_steps x emit_dim
    negCLL = tf.log(2 * np.pi) + tf.log(cov_p) + (x - mu_p) ** 2 / cov_p
    negCLL = negCLL * 0.5
    negCLL = tf.reduce_mean(negCLL, axis=0)  # sample
    negCLL = tf.reduce_sum(negCLL, axis=2)  # dim
    negCLL = tf.reduce_sum(negCLL, axis=1)  # T

    output_cost = {
        "cost": tf.reduce_mean(negCLL),
        "error": tf.reduce_mean(negCLL),
        "all_costs": tf.reduce_mean(negCLL),
    }
    # loss
    # outputs={
    # 	"sampled_pred_params":obs_list,
    # 	"sampled_z":zs}
    return output_cost
