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


#FLAGS = tf.app.flags.FLAGS


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
		#dtype = tf.half if FLAGS.use_fp16 else tf.float32
		dtype = tf.float32
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
	dtype = tf.float32
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

def dropout(data,drop_out_rate):
	return tf.nn.dropout(data,1.0-drop_out_rate)

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

def build_nn(x,dim_input,n_steps,hyparam_name,name,
	init_params_flag,
	params,
	control_params
		):
	wd_bias=None
	wd_w=0.1
	layer=x
	layer_dim=dim_input
	res_layer=None
	hy_param=hy.get_hyperparameter()
	for i,hy_layer in enumerate(hy_param[hyparam_name]):
		layer_dim_out=layer_dim
		#print(">>>",layer_dim)
		if "dim_output" in hy_layer:
			layer_dim_out=hy_layer["dim_output"]
			#print(">>>",layer_dim,"=>",layer_dim_out)
			
		if hy_layer["name"]=="fc":
			with tf.variable_scope(name+'_fc'+str(i)) as scope:
				layer=fc_layer("vd_fc"+str(i),layer,
					layer_dim, layer_dim_out,
					init_params_flag,params,wd_w,wd_bias,activate=tf.sigmoid)
				layer_dim=layer_dim_out
		elif hy_layer["name"]=="fc_res_start":
			res_layer=layer
		elif hy_layer["name"]=="do":
			layer=dropout(layer,control_params["dropout_rate"])
			layer_dim=layer_dim_out
		elif hy_layer["name"]=="fc_res":
			with tf.variable_scope(name+'_fc_res'+str(i)) as scope:
				layer=fc_layer(name+"_fc_res"+str(i),layer,
					layer_dim, layer_dim_out,
					init_params_flag,params,wd_w,wd_bias,activate=None)
				layer=res_layer+layer
				layer=tf.sigmoid(layer)
				layer_dim=layer_dim_out
		elif hy_layer["name"]=="fc_bn":
			with tf.variable_scope(name+'_fc_bn'+str(i)) as scope:
				layer=fc_layer(name+"_fcbn"+str(i),layer,
					layer_dim, layer_dim_out,
					init_params_flag,params,wd_w,wd_bias,activate=tf.sigmoid,with_bn=True)
				layer_dim=layer_dim_out
		elif hy_layer["name"]=="lstm":
			with tf.variable_scope(name+'_lstm'+str(i)) as scope:
				layer=tf.reshape(layer,[-1,n_steps,layer_dim])
				layer=RNN_layer(layer,n_steps,layer_dim_out)
				layer=tf.reshape(layer,[-1,layer_dim_out])
				layer_dim=layer_dim_out
		else:
			assert("unknown layer:"+hy_layer["name"])
	return layer,layer_dim
			
def computeVariationalDist(x,epsilon,n_steps,dim_emit,dim,control_params):
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
	##
	with tf.name_scope('variational_dist') as scope_parent:
		with tf.variable_scope('variational_dist_var') as v_scope_parent:
			layer,dim_out=build_nn(x,dim_input=dim_emit,n_steps=n_steps,
					hyparam_name="variational_internal_layers",name="vd",
					init_params_flag=init_params_flag,
					params=params,
					control_params=control_params
					)
			
			with tf.variable_scope('vd_fc_mu') as scope:
				layer_mu=fc_layer("vd_fc_mu",layer,
						dim_out, dim,
						init_params_flag,params,wd_w,wd_bias,
						activate=tf.tanh)
			with tf.variable_scope('vd_fc_cov') as scope:
				pre_activate=fc_layer("vd_fc_cov",layer,
						dim_out, dim,
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

def computeEmission(z,n_steps,dim,dim_emit,params=None,control_params=None):
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
		with tf.variable_scope('emission_var') as v_scope_parent:
			# z -> layer
			layer,dim_out=build_nn(z,dim_input=dim,n_steps=n_steps,
					hyparam_name="emission_internal_layers",name="em",
					init_params_flag=init_params_flag,
					params=params,
					control_params=control_params
					)
			#print(">>>>",dim_out)
			# layer -> layer_mean
			with tf.variable_scope('em_fc_mean') as scope:
				layer_mu=fc_layer("emission/em_fc2_mean",layer,
						dim_out, dim_emit,
						init_params_flag,params,wd_w,wd_bias,activate=None)
			# layer -> layer_cov
			with tf.variable_scope('em_fc_cov') as scope:
				pre_activate=fc_layer("emission/em_fc2_cov",layer,
						dim_out, dim_emit,
						init_params_flag,params,wd_w,wd_bias,activate=None)
				layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
	layer_mu=tf.reshape(layer_mu,[-1,n_steps,dim_emit])
	layer_cov=tf.reshape(layer_cov,[-1,n_steps,dim_emit])
	return [layer_mu,layer_cov],params
	
def computeTransitionOld(z,n_steps,dim,mean_prior0=None,cov_prior0=None,params=None,without_cov=False,control_params=None):
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
		with tf.variable_scope('transition_var') as v_scope_parent:
			# z -> layer
			if hy_param["transition_internal_layers"]:
				layer=build_nn(z,dim_input=dim,dim_output=dim,n_steps=n_steps,
					hyparam_name="transition_internal_layers",name="tr",
					init_params_flag=init_params_flag,
					params=params,
					control_params=control_params
					)
				# layer -> layer_mean
				with tf.variable_scope('tr_fc_mean') as scope:
					layer_mean=fc_layer("tr_fc_mean",layer,
							dim, dim,
							init_params_flag,params,wd_w,wd_bias,activate=None)
				if not without_cov:
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
	layer_cov=None
	if not without_cov:
		layer_cov =tf.reshape(layer_cov,[-1,n_steps,dim])
	
	if mean_prior0 is not None and cov_prior0 is not None:
		output_mu=tf.concat([mean_prior0,layer_mean[:,:-1,:]],axis=1)
		if not without_cov:
			output_cov =tf.concat([cov_prior0 ,layer_cov[:,:-1,:]],axis=1)
		else:
			output_cov=None
		return output_mu,output_cov,params
	else:
		return layer_mean,layer_cov,params
		
def computeTransitionFunc(in_points,n_steps,dim,params=None,control_params=None):
	hy_param=hy.get_hyperparameter()
	if hy_param["potential_grad_transition_enabled"]:
		return computeTransitionFuncFromPotential(in_points,n_steps,dim,params,control_params)
	else:
		return computeTransitionFuncFromNN(in_points,n_steps,dim,params,control_params)
	
def computeTransitionFuncFromNN(in_points,n_steps,dim,params=None,control_params=None):
	"""
	mu: points x dim
	"""
	if params is None:
		init_params_flag=True
		params={}
	else:
		init_params_flag=False
	
	wd_bias=None
	wd_w=0.1
	hy_param=hy.get_hyperparameter()
	z=tf.reshape(in_points,[-1,dim])
	with tf.name_scope('transition') as scope_parent:
		with tf.variable_scope('transition_var') as v_scope_parent:
			if hy_param["transition_internal_layers"]:
				# z -> layer
				layer,dim_out=build_nn(z,dim_input=dim,n_steps=n_steps,
					hyparam_name="transition_internal_layers",name="tr",
					init_params_flag=init_params_flag,
					params=params,
					control_params=control_params
					)
				# layer -> layer_mean
				with tf.variable_scope('tr_fc_mean') as scope:
					layer_mean=fc_layer("tr_fc_mean",layer,
							dim_out, dim,
							init_params_flag,params,wd_w,wd_bias,
							activate=tf.tanh)
			else:
				print("[INFO] default transition")
				layer=z
				layer_mean=fc_layer("tr_fc_mean",layer,
					dim, dim,
					init_params_flag,params,wd_w,wd_bias,activate=None)
	return layer_mean,params

def computeTransitionUKF(mu,cov,n_steps,dim,mean_prior0=None,cov_prior0=None,params=None,control_params=None):
	"""
	mu: bs x T x dim
	cov: bs x T x dim
	prior0: bs x 1 x dim
	"""
	#in_mu=tf.reshape(mu,[1,-1,dim])
	in_mu=tf.reshape(mu,[-1,dim])
	in_sigma=tf.sqrt(tf.reshape(cov,[-1,dim]))
	#in_points=tf.tile(in_mu,[dim*2+1,1,1])
	in_points=[in_mu]
	for i in range(dim):
		in_points.append(in_mu+in_sigma)
		in_points.append(in_mu-in_sigma)
		#in_points[i*2+0+1,:,:]+=in_sigma
		#in_points[i*2+1+1,:,:]-=in_sigma
	in_points=tf.stack(in_points)

	out_points,params=computeTransitionFunc(in_points,n_steps,dim,params=params,control_params=control_params)
	# N x bs x T x dim
	layer_mean=tf.reshape(out_points,[dim*2+1,-1,n_steps,dim])
	temp_sigma=layer_mean-layer_mean[0,:,:,:]
	layer_cov=tf.reduce_sum(temp_sigma**2,axis=0)

	if mean_prior0 is not None and cov_prior0 is not None:
		output_mu=tf.concat([mean_prior0,layer_mean[0,:,:-1,:]],axis=1)
		output_cov =tf.concat([cov_prior0 ,layer_cov[:,:-1,:]],axis=1)
		return output_mu,output_cov,params
	else:
		return layer_mean[0,:,:,:],layer_cov,params
		

def computePotential(z_input,n_steps,dim,params=None,control_params=None):
	hy_param=hy.get_hyperparameter()
	if hy_param["potential_nn_enabled"]:
		return computePotentialFromNN(z_input,n_steps,dim,params,control_params)
	else:
		return computePotentialWithBinaryPot(z_input,n_steps,dim,params,control_params)

def computePotentialWithBinaryPot(z_input,n_steps,dim,params=None,control_params=None):
	"""
	z: (bs x T) x dim
	pot: (bs x T)
	"""
	pot_pole=[]

	z=tf.reshape(z_input,[-1,dim])
	#z=tf.reshape(z_input,[900,dim])
	for d in range(dim):
		z1=np.zeros((dim,),dtype=np.float32)
		z2=np.zeros((dim,),dtype=np.float32)
		z1[d]=0.5
		z2[d]=-0.5
		
		z1=z-tf.constant(z1,dtype=np.float32)
		z2=z-tf.constant(z2,dtype=np.float32)
		p1=tf.reduce_sum(z1**2,axis=1)
		p2=tf.reduce_sum(z2**2,axis=1)
		pot_pole.append(p1)
		pot_pole.append(p2)
	pot_pole=tf.stack(pot_pole)
	#pot_pole: (2xdim) x (bs x T)
	pot=tf.reduce_min(pot_pole,axis=0)
	#z1=z+tf.constant([-1,0],dtype=np.float32)
	#z2=z+tf.constant([ 1,0],dtype=np.float32)
	#pot=tf.minimum(tf.reduce_sum(z1*z1,axis=2),tf.reduce_sum(z2*z2,axis=2))
	return pot,params


def computePotentialFromNN(z_input,n_steps,dim,params=None,control_params=None):
	"""
	z: (bs x T) x dim
	pot: (bs x T)
	"""
	if params is None:
		init_params_flag=True
		params={}
	else:
		init_params_flag=False
	
	wd_bias=None
	wd_w=0.1
	hy_param=hy.get_hyperparameter()
	z=tf.reshape(z_input,[-1,dim])
	with tf.name_scope('potential') as scope_parent:
		with tf.variable_scope('potential_var') as v_scope_parent:
			if hy_param["potential_internal_layers"]:
				# z -> layer
				layer,dim_out=build_nn(z,dim_input=dim,n_steps=n_steps,
					hyparam_name="potential_internal_layers",name="pot",
					init_params_flag=init_params_flag,
					params=params,
					control_params=control_params
					)
				# layer -> layer_mean
				with tf.variable_scope('pot_fc_mean') as scope:
					layer_mean=fc_layer("pot_fc_mean",layer,
							dim_out, 1,
							init_params_flag,params,wd_w,wd_bias,activate=tf.sigmoid)
			else:
				layer=z
				layer_mean=fc_layer("pot_fc_mean",layer,
					dim, 1,
					init_params_flag,params,wd_w,wd_bias,activate=tf.sigmoid)
	layer_mean=tf.reshape(layer_mean,[-1])
	return layer_mean,params


	if params is None:
		init_params_flag=True
		params={}
	else:
		init_params_flag=False
	wd_bias=None
	wd_w=0.1
	hy_param=hy.get_hyperparameter()
	pot_pole=[]

	z=tf.reshape(z_input,[-1,dim])
	for d in range(dim):
		z1=np.zeros((dim,),dtype=np.float32)
		z2=np.zeros((dim,),dtype=np.float32)
		z1[d]=1.0
		z2[d]=-1.0
		
		z1=z-tf.constant(z1,dtype=np.float32)
		z2=z-tf.constant(z2,dtype=np.float32)
		p1=tf.reduce_sum(z1*z1,axis=1)
		p2=tf.reduce_sum(z2*z2,axis=1)
		pot_pole.append(p1)
		pot_pole.append(p2)
		
	#pot_pole: (2xdim) x (bs x T)
	pot=tf.reduce_min(pot_pole,axis=0)
	#z1=z+tf.constant([-1,0],dtype=np.float32)
	#z2=z+tf.constant([ 1,0],dtype=np.float32)
	#pot=tf.minimum(tf.reduce_sum(z1*z1,axis=2),tf.reduce_sum(z2*z2,axis=2))
	return pot,params

def computeTransitionFuncFromPotential(in_points,n_steps,dim,params=None,control_params=None):
	"""
	in_points: points x dim
	"""
	with tf.name_scope('transition') as scope_parent:
		#z  : (bs x T) x dim
		#pot: (bs x T)
		pot,params=computePotential(in_points,n_steps,dim,params=params,control_params=control_params)
		sum_pot=tf.reduce_sum(pot)
		g_z = tf.gradients(sum_pot, [in_points])
		print(g_z)
		layer_mean=in_points+g_z
	
	return layer_mean,params
		

def p_filter(x,z,epsilon,dim,dim_emit,sample_size,batch_size,control_params):
	params_tr=None
	params_e=None
	proposal_sample_size=10
	resample_size=10
	#  x: (sample_size x batch_size) x dim_emit
	#  z: (sample_size x batch_size) x dim
	mu_trans,params_tr=computeTransitionFunc(z,1,dim,params=params_tr,control_params=control_params)
	#  m: sample_size x batch_size x dim
	m=tf.reshape(mu_trans,[sample_size,-1,dim])
	d=m - tf.reduce_mean(m,axis=0)
	cov=tf.reduce_mean(d**2,axis=0)
	cov_trans=tf.tile(cov,[sample_size,1])
	#  mu_trans : (sample_size x batch_size) x dim
	#  cov_trans: (sample_size x batch_size) x dim
	mu_trans=tf.reshape(mu_trans,[-1,dim])
	cov_trans=tf.reshape(cov_trans,[-1,dim])+0.01
	#mu_trans,cov_trans,params_tr=computeTransitionUKF(mu_q,cov_q,1,dim,None,None,param_tr)
	#
	proposal_dist=tf.contrib.distributions.Normal(mu_trans[:,:],cov_trans[:,:])
	particles=proposal_dist.sample(proposal_sample_size)
	#  particles: proposal_sample_size x (sample_size x batch_size) x dim
	#particles_d=particles-mu_trans[:,:]
	#particles_w=particles_d**2/cov_trans[:,:]
	#  particles: (proposal_sample_size x sample_size x batch_size) x dim
	particles =tf.reshape(particles,[-1,dim])
	obs_params,params_e=computeEmission(particles,1,dim,dim_emit,params_e,control_params=control_params)
	#  mu: (proposal_sample_size x sample_size)  x batch_size x emit_dim
	#  cov: (proposal_sample_size x sample_size) x batch_size x emit_dim
	mu=obs_params[0]
	cov=obs_params[1]
	mu =tf.reshape(mu,[-1,batch_size,dim_emit])
	cov =tf.reshape(cov,[-1,batch_size,dim_emit])
	#  x: batch_size x emit_dim
	d=mu-x[:,:]
	w=-tf.reduce_sum(d**2/cov,axis=2)-100
	# w:(proposal_sample_size x sample_size) x batch__size 
	#  probs=w/tf.reduce_sum(w,axis=0)

	resample_dist = tf.contrib.distributions.Categorical(logits=tf.transpose(w))
	# ids: resample x batch_size
	#  particles: (proposal_sample_size x sample_size) x batch_size x dim
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




def p_filter2(x,z,step,epsilon,n_steps,dim,dim_emit,batch_size,control_params):
	params_tr=None
	params_e=None
	sample_size=10
	resample_size=10
	#  z0: (in_sample x b) x dim
	#mu_trans,cov_trans,params_tr=computeTransition(z,1,dim,None,None,params_tr,without_cov=True,control_params=control_params)
	mu_trans,params_tr=computeTransitionFunc(z,1,dim,params=params_tr,control_params=control_params)
	#  m: Sample x (in_sample x b) x 1 x dim
	m=tf.reshape(mu_trans,[sample_size,-1,1,dim])
	d=m - tf.reduce_mean(m,axis=0)
	cov=tf.reduce_mean(d**2,axis=0)
	cov_trans=tf.tile(cov,[sample_size,1,1])
	m=tf.reshape(mu_trans,[-1,1,dim])
	#mu_trans,cov_trans,params_tr=computeTransitionUKF(mu_q,cov_q,1,dim,None,None,param_tr)
	#
	dist=tf.contrib.distributions.Normal(m[:,0,:],cov_trans[:,0,:])
	particles=dist.sample(sample_size)
	#  particles: Sample x (in_sample x b) x dim
	particles_d=particles-m[:,0,:]
	particles_w=particles_d**2/cov_trans[:,0,:]
	particles =tf.reshape(particles,[-1,dim])
	obs_params,params_e=computeEmission(particles,1,dim,dim_emit,params_e,control_params=control_params)
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

def inference(x,epsilon,n_steps,dim,dim_emit,pot_points=None,control_params=None):
	"""
	Returns:
		Logits.
	"""
	# We instantiate all variables using tf.get_variable() instead of
	# tf.Variable() in order to share variables across multiple GPU training runs.
	# If we only ran this model on a single GPU, we could simplify this function
	# by replacing all instances of tf.get_variable() with tf.Variable().
	#
	hy_param=hy.get_hyperparameter()
	with tf.name_scope('inference') as scope_parent:
		z_q,mu_q,cov_q=computeVariationalDist(x,epsilon,n_steps,dim_emit,dim,control_params=control_params)
		m0=tf.constant(np.zeros((1,1,dim),dtype=np.float32))
		c0=tf.constant(np.ones((1,1,dim),dtype=np.float32))
		bs=tf.shape(mu_q)[0]
		mean_prior0=tf.tile(m0,(bs,1,1))
		cov_prior0 =tf.tile(c0,(bs,1,1))
		#mu_trans,cov_trans,params_tr=computeTransition(z_q,n_steps,dim,mean_prior0,cov_prior0)
		mu_trans,cov_trans,params_tr=computeTransitionUKF(mu_q,cov_q,n_steps,dim,mean_prior0,cov_prior0,control_params=control_params)
		pot_loss=None
		if hy_param["potential_enabled"]:
			if hy_param["potential_grad_transition_enabled"]==False:
				use_data_points=False
				if pot_points is None:
					use_data_points=True
				if use_data_points:
					## compute V(x(t+1))-V(x(t)) < 0 for stability
					mu_trans_1,cov_trans_1,params_tr=computeTransitionUKF(mu_q,cov_q,n_steps,dim,None,None,params_tr,control_params=control_params)
					#mu_q: bs x T x dim
					#mu_trans_1: bs x T x dim
					params_pot=None
					pot0,params_pot=computePotential(mu_q,n_steps,dim,params=params_pot,control_params=control_params)
					pot1,params_pot=computePotential(mu_trans_1,n_steps,dim,params=params_pot,control_params=control_params)
					#pot: bs x T
					c=0.1
					pot=tf.nn.relu(pot1-pot0+c)
					pot_loss=tf.reshape(pot,[-1,n_steps])
				else:
					mu_trans_1,params_tr=computeTransitionFunc(pot_points,1,dim,params=params_tr,control_params=control_params)
					params_pot=None
					pot0,params_pot=computePotential(pot_points,1,dim,params=params_pot,control_params=control_params)
					pot1,params_pot=computePotential(mu_trans_1,1,dim,params=params_pot,control_params=control_params)
					#pot: bs x T
					c=0.1
					pot=tf.nn.relu(pot1-pot0+c)
					pot_loss=tf.reshape(pot,[-1,1])
		# compute emission
		obs_params,params=computeEmission(z_q,n_steps,dim,dim_emit,control_params=control_params)
		mu_trans_input=tf.reshape(mu_trans,[-1,dim])
		pred_params,_=computeEmission(mu_trans_input,n_steps,dim,dim_emit,params,control_params=control_params)
		#_activation_summary(softmax_linear)

	outputs={"mu_q":mu_q,"cov_q":cov_q,
			"mu_tr":mu_trans,"cov_tr":cov_trans,
			"obs_params":obs_params,"pred_params":pred_params,"z_q":z_q,"potential_loss":pot_loss}
	return outputs

def computeNegCLL(x,outputs,mask):
	eps=1.0e-10
	cov_p=outputs["obs_params"][1]+eps
	mu_p=outputs["obs_params"][0]

	negCLL=tf.log(2*np.pi)+tf.log(cov_p)+(x-mu_p)**2/cov_p
	masked_negCLL=tf.reduce_sum(negCLL*0.5*mask,axis=2)
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
	masked_kl=tf.reduce_sum(kl_t,axis=2)
	return tf.reduce_sum(masked_kl,axis=1)

def loss(x,outputs,mask,alpha=1,control_params=None):
	"""Add 
	Returns:
		Loss tensor of type float.
	"""
	negCLL=computeNegCLL(x,outputs,mask)
	temporalKL=computeTemporalKL(x,outputs,mask)
	cost_pot=tf.constant(0.0,dtype=np.float32)
	if outputs["potential_loss"] is not None:
		pot=outputs["potential_loss"]
		#sum_pot=tf.reduce_sum(pot*mask,axis=1)
		sum_pot=tf.reduce_sum(pot,axis=1)
		cost_pot=tf.reduce_mean(pot)
	cost_mean = tf.reduce_mean(negCLL+alpha*temporalKL+alpha*1.0*cost_pot, name='train_cost')
	#cost_mean = tf.reduce_mean((1-alpha)*negCLL+alpha*temporalKL+alpha*1.0*cost_pot, name='train_cost')
	tf.add_to_collection('losses', cost_mean)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	total_cost=tf.add_n(tf.get_collection('losses'), name='total_loss')
	#costs={"negCLL":negCLL,"temporalKL":temporalKL,"potentialCost":cost_pot}
	costs=[tf.reduce_mean(negCLL),tf.reduce_mean(temporalKL),cost_pot]
	return total_cost,cost_mean,costs


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

