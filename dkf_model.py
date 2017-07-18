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


FLAGS = tf.app.flags.FLAGS


# Basic model parameters.
#tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")


# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
	"""Helper to create summaries for activations.

	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.

	Args:
		x: Tensor
	Returns:
		nothing
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.summary.histogram(tensor_name + '/activations', x)
	tf.summary.scalar(tensor_name + '/sparsity',
						tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
	"""Helper to create a Variable stored on CPU memory.

	Args:
		name: name of the variable
		shape: list of ints
		initializer: initializer for Variable

	Returns:
		Variable Tensor
	"""
	with tf.device('/cpu:0'):
		dtype = tf.half if FLAGS.use_fp16 else tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
	return var


def _variable_with_weight_decay(name, shape,initializer_name, wd):
	"""Helper to create an initialized Variable with weight decay.

	Note that the Variable is initialized with a truncated normal distribution.
	A weight decay is added only if one is specified.

	Args:
		name: name of the variable
		shape: list of ints
		stddev: standard deviation of a truncated Gaussian
		wd: add L2Loss weight decay multiplied by this float. If None, weight
				decay is not added for this Variable.

	Returns:
		Variable Tensor
	"""
	dtype = tf.half if FLAGS.use_fp16 else tf.float32
	if initializer_name=="normal":
		stddev=1e-5
		var = _variable_on_cpu(name,shape,
				tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
	elif initializer_name=="zero":
		var = _variable_on_cpu(name, shape,
			tf.constant_initializer(0.0,dtype=dtype))
	if wd is not None:  
		weight_decay = tf.nn.l2_loss(var)* wd
		tf.add_to_collection('losses', weight_decay)
	return var

def _normalization(data,name):
	bias=1.0
	alpha=0.001 / 9.0
	beta=0.75
	output = tf.nn.lrn(data, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
	return output

def batch_normalization(name,dim, data,init_params_flag,params):
	if init_params_flag:
		gamma = _variable_with_weight_decay(
			'gamma',
			shape=[dim],
			initializer_name="normal",
			wd=None)
		beta  = _variable_with_weight_decay(
			'beta',
			shape=[dim],
			initializer_name="normal",
			wd=None)
		if params is not None:
			params[name+"/gamma"]=gamma
			params[name+"/beta"]=beta
	else:
		gamma=params[name+"/gamma"]
		beta=params[name+"/beta"]

	eps = 1e-5
	mean, variance = tf.nn.moments(data, [0])
	return gamma * (data - mean) / tf.sqrt(variance + eps) + beta

def RNN_layer(x,n_steps,n_output):
	# x: (batch_size, n_steps, n_input)
	
	# a list of 'n_steps' tensors of shape (batch_size, n_input)
	x = tf.unstack(x, n_steps, axis=1)
	# Define a lstm cell with tensorflow
	lstm_cell = tf.contrib.rnn.LSTMCell(
			n_output, forget_bias=1.0,activation=tf.tanh,
			initializer=tf.constant_initializer(0.0))
	outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
	# 'outputs' is a list of output tensor of shape (batch_size, n_output)
	# and change back dimension to (batch_size, n_step, n_output)
	outputs = tf.stack(outputs)
	outputs = tf.transpose(outputs, [1, 0, 2])
	return outputs

def fc_layer(name,input_layer,dim_in, dim_out,init_params_flag,params,wd_w,wd_b,activate,with_bn=False):
	if init_params_flag:
		w = _variable_with_weight_decay('weights', [dim_in, dim_out],
			initializer_name="normal", wd=wd_w)
		b = _variable_with_weight_decay('biases', [dim_out],
			initializer_name="zero",wd=wd_b)
		if params is not None:
			params[name+"/w"]=w
			params[name+"/b"]=b
	else:
		w=params[name+"/w"]
		b=params[name+"/b"]
	pre_activate=tf.nn.bias_add(tf.matmul(input_layer,w),b)
	# bn
	if with_bn:
		pre_activate=batch_normalization(name,dim_out, pre_activate,init_params_flag,params)
	#
	if activate is None:
		layer = pre_activate
	else:
		layer = tf.sigmoid(pre_activate)
	return layer



def computeVariationalDist(x,epsilon,n_steps,dim_emit,dim):
	"""
	x: bs x T x dim_emit
	return:
		layer_z:bs x T x dim
		layer_mu:bs x T x dim
		layer_cov:bs x T x dim
	"""
	wd_bias=None
	wd_w=0.1
	x=tf.reshape(x,[-1,dim_emit])
	init_params_flag=True
	params=None
	hy_param=hy.get_hyperparameter()
	with tf.name_scope('variational_dist') as scope_parent:
		layer=x
		res_layer=None
		for i,hy_layer in enumerate(hy_param["variational_internal_layers"]):
			if hy_layer["name"]=="fc":
				with tf.variable_scope('vd_fc'+str(i)) as scope:
					layer=fc_layer("vd_fc"+str(i),layer,
						dim_emit, dim_emit,
						init_params_flag,params,wd_w,wd_bias,activate=tf.sigmoid)
			elif hy_layer["name"]=="fc_res_start":
				res_layer=layer
			elif hy_layer["name"]=="fc_res":
				with tf.variable_scope('vd_fc_res'+str(i)) as scope:
					layer=fc_layer("vd_fc_res"+str(i),layer,
						dim_emit, dim_emit,
						init_params_flag,params,wd_w,wd_bias,activate=None)
					layer=res_layer+layer
					layer=tf.sigmoid(layer)
			elif hy_layer["name"]=="fc_bn":
				with tf.variable_scope('vd_fc_bn'+str(i)) as scope:
					layer=fc_layer("vd_fc"+str(i),layer,
						dim_emit, dim_emit,
						init_params_flag,params,wd_w,wd_bias,activate=tf.sigmoid,with_bn=True)
			elif hy_layer["name"]=="lstm":
				with tf.variable_scope('vd_lstm'+str(i)) as scope:
					layer=tf.reshape(layer,[-1,n_steps,dim_emit])
					layer=RNN_layer(layer,n_steps,dim_emit)
					layer=tf.reshape(layer,[-1,dim_emit])
			else:
				assert("unknown layer:"+hy_layer["name"])
			
		with tf.variable_scope('vd_fc_mu') as scope:
			layer_mu=fc_layer("vd_fc_mu",layer,
					dim_emit, dim,
					init_params_flag,params,wd_w,wd_bias,activate=None)
		with tf.variable_scope('vd_fc_cov') as scope:
			pre_activate=fc_layer("vd_fc_cov",layer,
					dim_emit, dim,
					init_params_flag,params,wd_w,wd_bias,activate=None)
			layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
		with tf.variable_scope('vd_fc_z')	 as scope:
			if epsilon is not None:
				layer_z=layer_mu+tf.sqrt(layer_cov)*epsilon
			else:
				layer_z=layer_mu

		layer_z=tf.reshape(layer_z,[-1,dim])
		layer_mu=tf.reshape(layer_mu,[-1,n_steps,dim])
		layer_cov=tf.reshape(layer_cov,[-1,n_steps,dim])
	return layer_z,layer_mu,layer_cov

def computeEmission(z,n_steps,dim,dim_emit,params=None):
	"""
	z: (bs x T) x dim
	"""
	if params is None:
		init_params_flag=True
		params={}
	else:
		init_params_flag=False

	wd_bias=None
	wd_w=0.1
	hy_param=hy.get_hyperparameter()
	with tf.name_scope('emission') as scope_parent:
		# z -> layer
		layer=z
		res_layer=None
		for i,hy_layer in enumerate(hy_param["emssion_internal_layers"]):
			
			if hy_layer["name"]=="fc_bn":
				with tf.variable_scope('em_fc_bn'+str(i)) as scope:
					layer=fc_layer("em_fc_bn"+str(i),layer,
						dim, dim,
						init_params_flag,params,wd_w,wd_bias,activate=tf.sigmoid,with_bn=True)
			elif hy_layer["name"]=="fc":
				with tf.variable_scope('em_fc'+str(i)) as scope:
					layer=fc_layer("em_fc"+str(i),layer,
						dim, dim,
						init_params_flag,params,wd_w,wd_bias,activate=tf.sigmoid)
			elif hy_layer["name"]=="fc_res_start":
				res_layer=layer
			elif hy_layer["name"]=="fc_res":
				with tf.variable_scope('em_fc_res'+str(i)) as scope:
					layer=fc_layer("em_fc_res"+str(i),layer,
						dim, dim,
						init_params_flag,params,wd_w,wd_bias,activate=None)
					layer=res_layer+layer
					layer=tf.sigmoid(layer)

		# layer -> layer_mean
		with tf.variable_scope('em_fc_mean') as scope:
			layer_mu=fc_layer("emission/em_fc2_mean",z,
					dim, dim_emit,
					init_params_flag,params,wd_w,wd_bias,activate=None)
		# layer -> layer_cov
		with tf.variable_scope('em_fc_cov') as scope:
			pre_activate=fc_layer("emission/em_fc2_cov",z,
					dim, dim_emit,
					init_params_flag,params,wd_w,wd_bias,activate=None)
			layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
	layer_mu=tf.reshape(layer_mu,[-1,n_steps,dim_emit])
	layer_cov=tf.reshape(layer_cov,[-1,n_steps,dim_emit])
	return [layer_mu,layer_cov],params


def computeTransition(z,n_steps,dim,mean_prior0=None,cov_prior0=None,params=None):
	"""
	z: (bs x T)x dim
	prior0: bs x 1 x dim
	"""
	if params is None:
		init_params_flag=True
		params={}
	else:
		init_params_flag=False


	wd_bias=None
	wd_w=0.1
	hy_param=hy.get_hyperparameter()
	with tf.name_scope('transition') as scope_parent:
		# z -> layer
		layer=z
		if hy_param["transition_internal_layers"]:
			for i,hy_layer in enumerate(hy_param["transition_internal_layers"]):
				if hy_layer["name"]=="fc":
					with tf.variable_scope('tr_fc1'+str(i)) as scope:
						layer=fc_layer("tr_fc"+str(i),layer,
							dim, dim,
							init_params_flag,params,wd_w,wd_bias,activate=tf.sigmoid)
			# layer -> layer_mean
			with tf.variable_scope('tr_fc_mean') as scope:
				layer_mean=fc_layer("tr_fc_mean",layer,
						dim, dim,
						init_params_flag,params,wd_w,wd_bias,activate=None)
			# layer -> layer_cov
			with tf.variable_scope('tr_fc_cov') as scope:
				pre_activate=fc_layer("tr_fc_cov",layer,
						dim, dim,
						init_params_flag,params,wd_w,wd_bias,activate=None)
				layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
		else:
			layer=z
			layer_mean=fc_layer("tr_fc_mean",layer,
				dim, dim,
				init_params_flag,params,wd_w,wd_bias,activate=None)

	# bs x T x dim
	layer_mean=tf.reshape(layer_mean,[-1,n_steps,dim])
	layer_cov =tf.reshape(layer_cov,[-1,n_steps,dim])
	if mean_prior0 is not None and cov_prior0 is not None:
		output_mu=tf.concat([mean_prior0,layer_mean[:,:-1,:]],axis=1)
		output_cov =tf.concat([cov_prior0 ,layer_cov[:,:-1,:]],axis=1)
		return output_mu,output_cov,params
	else:
		return layer_mean,layer_cov,params
		


def p_filter(x,z,step,epsilon,n_steps,dim,dim_emit,batch_size):
	params_tr=None
	params_e=None
	sample_size=10
	resample_size=10
	#  z0: (in_sample x b) x dim
	mu_trans,cov_trans,params_tr=computeTransition(z,1,dim,None,None,params_tr)
	dist=tf.contrib.distributions.Normal(mu_trans[:,0,:],cov_trans[:,0,:])
	particles=dist.sample(sample_size)
	#  particles: Sample x (in_sample x b) x dim
	particles_d=particles-mu_trans[:,0,:]
	particles_w=particles_d**2/cov_trans[:,0,:]
	particles =tf.reshape(particles,[-1,dim])
	obs_params,params_e=computeEmission(particles,1,dim,dim_emit,params_e)
	mu=obs_params[0]
	cov=obs_params[1]
	mu =tf.reshape(mu,[-1,batch_size,dim_emit])
	cov =tf.reshape(cov,[-1,batch_size,dim_emit])
	d=mu-x[:,step,:]
	w=-tf.reduce_sum(d**2/cov,axis=2)
	# w: (Sample x in_sample) x b 
	#probs=w/tf.reduce_sum(w,axis=0)

	resample_dist = tf.contrib.distributions.Categorical(logits=tf.transpose(w))
	# ids: Resample x b 
	particle_ids=resample_dist.sample([resample_size])
	particles =tf.reshape(particles,[-1,batch_size,dim])
	#
	dummy=np.zeros((resample_size,batch_size,1),dtype=np.int32)
	particle_ids=tf.reshape(particle_ids,[resample_size,batch_size,1])
	for i in range(batch_size):
		dummy[:,i,0]=i
	temp=tf.constant(dummy)
	particle_ids=tf.concat([particle_ids, temp], 2)
	# particles: (Sample x in_sample) x b x dim
	out=tf.gather_nd(particles,particle_ids)

	outputs={"sampled_pred_params":[mu,cov],
			"sampled_z":out}
	return outputs

def inference(x,epsilon,n_steps,dim,dim_emit):
	"""
	Returns:
		Logits.
	"""
	# We instantiate all variables using tf.get_variable() instead of
	# tf.Variable() in order to share variables across multiple GPU training runs.
	# If we only ran this model on a single GPU, we could simplify this function
	# by replacing all instances of tf.get_variable() with tf.Variable().
	#
	with tf.name_scope('inference') as scope_parent:
		z_q,mu_q,cov_q=computeVariationalDist(x,epsilon,n_steps,dim_emit,dim)
		m0=tf.constant(np.zeros((1,1,dim),dtype=np.float32))
		c0=tf.constant(np.ones((1,1,dim),dtype=np.float32))
		bs=tf.shape(mu_q)[0]
		mean_prior0=tf.tile(m0,(bs,1,1))
		cov_prior0 =tf.tile(c0,(bs,1,1))
		mu_trans,cov_trans,_=computeTransition(z_q,n_steps,dim,mean_prior0,cov_prior0)
		obs_params,params=computeEmission(z_q,n_steps,dim,dim_emit)
		mu_trans_input=tf.reshape(mu_trans,[-1,dim])
		pred_params,_=computeEmission(mu_trans_input,n_steps,dim,dim_emit,params)
		#_activation_summary(softmax_linear)

	outputs={"mu_q":mu_q,"cov_q":cov_q,
			"mu_tr":mu_trans,"cov_tr":cov_trans,
			"obs_params":obs_params,"pred_params":pred_params,"z_q":z_q}
	return outputs

def computeNegCLL(x,outputs,mask):
	eps=1.0e-10
	cov_p=outputs["obs_params"][1]+eps
	mu_p=outputs["obs_params"][0]

	negCLL=tf.log(2*np.pi)+tf.log(cov_p)+(x-mu_p)**2/cov_p
	masked_negCLL=tf.reduce_sum(negCLL*0.5,axis=2)*mask
	return tf.reduce_sum(masked_negCLL,axis=1)

def computeTemporalKL(x,outputs,mask):
	"""
	z: bs x T x dim
	prior0: bs x 1 x dim
	"""
	eps=1.0e-10
	cov_p=outputs["cov_tr"]+eps
	mu_p=outputs["mu_tr"]
	cov_q=outputs["cov_q"]+eps
	mu_q=outputs["mu_q"]
	#cov_p=outputs["cov_tr"][:,:-1,:]+eps
	#mu_p=outputs["mu_tr"][:,:-1,:]
	#cov_q=outputs["cov_q"][:,1:,:]+eps
	#mu_q=outputs["mu_q"][:,1:,:]
	

	kl_t=tf.log(cov_p)-tf.log(cov_q)-1+cov_q/cov_p+(mu_p-mu_q)**2/cov_p
	masked_kl=tf.reduce_sum(kl_t,axis=2)*mask
	return tf.reduce_sum(masked_kl,axis=1)

def loss(x,outputs,mask,alpha=1):
	"""Add 
	Returns:
		Loss tensor of type float.
	"""
	negCLL=computeNegCLL(x,outputs,mask)
	temporalKL=computeTemporalKL(x,outputs,mask)
	
	cost_mean = tf.reduce_mean(negCLL+alpha*temporalKL, name='train_cost')
	tf.add_to_collection('losses', cost_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	total_cost=tf.add_n(tf.get_collection('losses'), name='total_loss')

	return total_cost,cost_mean,negCLL,temporalKL


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
	loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
	losses = tf.get_collection('losses')
	loss_averages_op = loss_averages.apply(losses + [total_loss])

	# Attach a scalar summary to all individual losses and the total loss; do the
	# same for the averaged version of the losses.
	for l in losses + [total_loss]:
		# Name each loss as '(raw)' and name the moving average version of the loss
		# as the original loss name.
		tf.summary.scalar(l.op.name + ' (raw)', l)
		tf.summary.scalar(l.op.name, loss_averages.average(l))

	return loss_averages_op

