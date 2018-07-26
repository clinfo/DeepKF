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

#FLAGS = tf.app.flags.FLAGS

def construct_placeholder(config):
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	dim_emit=hy_param["dim_emit"]
	n_steps=hy_param["n_steps"]
	# 
	x_holder=tf.placeholder(tf.float32,shape=(None,n_steps,dim_emit))
	m_holder=tf.placeholder(tf.float32,shape=(None,n_steps,dim_emit))
	s_holder=tf.placeholder(tf.int32,shape=(None,))
	vd_eps_holder=tf.placeholder(tf.float32,shape=(None,n_steps,dim))
	tr_eps_holder=tf.placeholder(tf.float32,shape=(None,n_steps,dim))
	potential_points_holder=tf.placeholder(tf.float32,shape=(None,dim))
	alpha_holder=tf.placeholder(tf.float32)
	dropout_rate=tf.placeholder(tf.float32)
	is_train=tf.placeholder(tf.bool)
	#
	placeholders={"x":x_holder,
			"m":m_holder,
			"s":s_holder,
			"potential_points": potential_points_holder,
			"alpha": alpha_holder,
			"vd_eps": vd_eps_holder,
			"tr_eps": tr_eps_holder,
			"dropout_rate": dropout_rate,
			"is_train": is_train,
			}
	return placeholders


def build_nn(x,dim_input,n_steps,hyparam_name,name,
		init_params_flag,
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
				layer=layers.fc_layer("vd_fc"+str(i),layer,
					layer_dim, layer_dim_out,
					wd_w,wd_bias,activate=tf.sigmoid,init_params_flag=init_params_flag)
				layer_dim=layer_dim_out
		elif hy_layer["name"]=="fc_res_start":
			res_layer=layer
		elif hy_layer["name"]=="do":
			dropout_rate=control_params["placeholders"]["dropout_rate"]
			layer=layers.dropout_layer(layer,dropout_rate)
			layer_dim=layer_dim_out
		elif hy_layer["name"]=="fc_res":
			with tf.variable_scope(name+'_fc_res'+str(i)) as scope:
				layer=layers.fc_layer(name+"_fc_res"+str(i),layer,
					layer_dim, layer_dim_out,
					wd_w,wd_bias,activate=None,init_params_flag=init_params_flag)
				layer=res_layer+layer
				layer=tf.sigmoid(layer)
				layer_dim=layer_dim_out
		elif hy_layer["name"]=="fc_bn":
			is_train=control_params["placeholders"]["is_train"]
			with tf.variable_scope(name+'_fc_bn'+str(i)) as scope:
				layer=layers.fc_layer(name+"_fcbn"+str(i),layer,
					layer_dim, layer_dim_out,
					wd_w,wd_bias,activate=tf.sigmoid,with_bn=True,init_params_flag=init_params_flag,is_train=is_train)
				layer_dim=layer_dim_out
		elif hy_layer["name"]=="lstm":
			with tf.variable_scope(name+'_lstm'+str(i)) as scope:
				layer=tf.reshape(layer,[-1,n_steps,layer_dim])
				layer=layers.lstm_layer(layer,n_steps,layer_dim_out,init_params_flag=init_params_flag)
				layer=tf.reshape(layer,[-1,layer_dim_out])
				layer_dim=layer_dim_out
		else:
			assert("unknown layer:"+hy_layer["name"])
	return layer,layer_dim

def sample_normal(params,eps):
	mu=params[0]
	cov=params[1]
	if eps is not None:
		return mu+tf.sqrt(cov)*eps
	else:
		return mu

# return sample z from q(z): tensors: batch_size x n_steps x dim
def sampleVariationalDist(x,n_steps,init_params_flag=True,control_params=None):
	if control_params["state_type"]=="discrete":
		qz=computeVariationalDist(x,n_steps,init_params_flag,control_params)
		if control_params["sampling_type"]=="none":
			z_s=qz[0]
		elif control_params["sampling_type"]=="gambel-max":
			eps=control_params["placeholders"]["vd_eps"]
			g=eps
			#g=-tf.log(-tf.log(eps))
			logpi=tf.log(qz[0]+1.0e-10)
			dist=tf.contrib.distributions.OneHotCategorical(logit=(logpi+g))
			z_s=tf.cast(dist.sample(),tf.float32)
		elif control_params["sampling_type"]=="gambel-softmax":
			eps=control_params["placeholders"]["vd_eps"]
			g=eps
			#g=-tf.log(-tf.log(eps))
			logpi=tf.log(qz[0]+1.0e-5)
			tau=10.0
			z_s=tf.nn.softmax((logpi+g)/tau)
		elif control_params["sampling_type"]=="naive":
			dist=tf.contrib.distributions.OneHotCategorical(probs=qz[0])
			z_s=tf.cast(dist.sample(),tf.float32)
		else:
			raise Exception('[Error] unknown sampling type')
	elif control_params["state_type"]=="normal":
		qz=computeVariationalDist(x,n_steps,init_params_flag,control_params)
		if control_params["sampling_type"]=="none":
			z_s=qz[0]
		elif control_params["sampling_type"]=="normal":
			eps=control_params["placeholders"]["vd_eps"]
			z_s=sample_normal(qz,eps)
		else:
			raise Exception('[Error] unknown sampling type')
	else:
		raise Exception('[Error] unknown state type')
	return z_s,qz

# return parameters for q(z): list of tensors: batch_size x n_steps x dim
def computeVariationalDist(x,n_steps,init_params_flag=True,control_params=None):
	"""
	x: bs x T x dim_emit
	return:
		layer_z:bs x T x dim
		layer_mu:bs x T x dim
		layer_cov:bs x T x dim
	"""
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	dim_emit=hy_param["dim_emit"]
	wd_bias=None
	wd_w=0.1
	x=tf.reshape(x,[-1,dim_emit])
	##
	params=[]
	with tf.name_scope('variational_dist') as scope_parent:
		with tf.variable_scope('variational_dist_var') as v_scope_parent:
			layer,dim_out=build_nn(x,dim_input=dim_emit,n_steps=n_steps,
					hyparam_name="variational_internal_layers",name="vd",
					init_params_flag=init_params_flag,
					control_params=control_params
					)
			if control_params["state_type"]=="discrete":
				with tf.variable_scope('vd_fc_logits') as scope:
					layer_logit=layers.fc_layer("vd_fc_logits",layer,
							dim_out, dim,
							wd_w,wd_bias,
							activate=tf.tanh,init_params_flag=init_params_flag)

				layer_logit=tf.reshape(layer_logit,[-1,n_steps,dim])
				layer_z=tf.nn.softmax(layer_logit)
				params.append(layer_z)
			elif control_params["state_type"]=="normal":
				with tf.variable_scope('vd_fc_mu') as scope:
					layer_mu=layers.fc_layer("vd_fc_mu",layer,
							dim_out, dim,
							wd_w,wd_bias,
							activate=tf.tanh,init_params_flag=init_params_flag)
					layer_mu=tf.reshape(layer_mu,[-1,n_steps,dim])
					params.append(layer_mu)
				with tf.variable_scope('vd_fc_cov') as scope:
					pre_activate=layers.fc_layer("vd_fc_cov",layer,
							dim_out, dim,
							wd_w,wd_bias,
							activate=None,init_params_flag=init_params_flag)
					layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
					layer_cov=tf.reshape(layer_cov,[-1,n_steps,dim])
					params.append(layer_cov)
			else:
				raise Exception('[Error] unknown state type')

	return params

# return parameters for p(x|z): list of tensors: batch_size x n_steps x dim
def computeEmission(z,n_steps,init_params_flag=True,control_params=None):
	"""
	z: (bs x T) x dim
	"""

	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	dim_emit=hy_param["dim_emit"]
	wd_bias=None
	wd_w=0.1
	hy_param=hy.get_hyperparameter()
	z=tf.reshape(z,[-1,dim])
	params=[]
	with tf.name_scope('emission') as scope_parent:
		with tf.variable_scope('emission_var') as v_scope_parent:
			# z -> layer
			layer,dim_out=build_nn(z,dim_input=dim,n_steps=n_steps,
					hyparam_name="emission_internal_layers",name="em",
					init_params_flag=init_params_flag,
					control_params=control_params
					)
			if control_params["emission_type"]=="normal":
				# layer -> layer_mean
				with tf.variable_scope('em_fc_mean') as scope:
					layer_mu=layers.fc_layer("emission/em_fc2_mean",layer,
							dim_out, dim_emit,
							wd_w,wd_bias,activate=None,init_params_flag=init_params_flag)
				layer_mu=tf.reshape(layer_mu,[-1,n_steps,dim_emit])
				params.append(layer_mu)
				# layer -> layer_cov
				with tf.variable_scope('em_fc_cov') as scope:
					pre_activate=layers.fc_layer("emission/em_fc2_cov",layer,
							dim_out, dim_emit,
							wd_w,wd_bias,activate=None,init_params_flag=init_params_flag)
					layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
				layer_cov=tf.reshape(layer_cov,[-1,n_steps,dim_emit])
				params.append(layer_cov)
			elif control_params["emission_type"]=="binary":
				# layer -> sigmoid
				with tf.variable_scope('em_fc_out') as scope:
					layer_logit=layers.fc_layer("emission/em_fc_out",layer,
							dim_out, dim,
							wd_w,wd_bias,
							activate=None,init_params_flag=init_params_flag)
					layer_out=tf.nn.sigmoid(layer_logit)
				layer_out=tf.reshape(layer_out,[-1,n_steps,dim_emit])
				params.append(layer_out)
			else:
				raise Exception('[Error] unknown emmition type')


	return params

# return parameters for p(z_t|z_t-1): list of tensors: batch_size x n_steps x dim
def computeTransitionFunc(in_points,n_steps,init_params_flag=True,control_params=None):
	hy_param=hy.get_hyperparameter()
	if hy_param["potential_grad_transition_enabled"]:
		return computeTransitionFuncFromPotential(in_points,n_steps,init_params_flag,control_params)
	else:
		return computeTransitionFuncFromNN(in_points,n_steps,init_params_flag,control_params)
	
# p(z_t|z_t+1)
def computeTransitionDistWithNN(in_points,n_steps,init_params_flag=True,control_params=None):
	"""
	mu: points x dim
	"""
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	
	wd_bias=None
	wd_w=0.1
	hy_param=hy.get_hyperparameter()
	z=tf.reshape(in_points,[-1,dim])
	params=[]
	with tf.name_scope('transition') as scope_parent:
		with tf.variable_scope('transition_var') as v_scope_parent:
			layer=z
			if hy_param["transition_internal_layers"]:
				# z -> layer
				layer,dim_out=build_nn(layer,dim_input=dim,n_steps=n_steps,
					hyparam_name="transition_internal_layers",name="tr",
					init_params_flag=init_params_flag,
					control_params=control_params
					)
			if control_params["state_type"]=="discrete":
				# layer -> layer_mean
				with tf.variable_scope('tr_fc_logits') as scope:
					layer_logit=layers.fc_layer("tr_fc_logits",layer,
							dim_out, dim,
							wd_w,wd_bias,
							activate=None,init_params_flag=init_params_flag)
					layer_z=tf.nn.softmax(layer_logit)
					layer_z=tf.reshape(layer_z,[-1,n_steps,dim])
					params.append(layer_z)
			elif control_params["state_type"]=="normal":
				with tf.variable_scope('vd_fc_mu') as scope:
					layer_mu=layers.fc_layer("vd_fc_mu",layer,
							dim_out, dim,
							wd_w,wd_bias,
							activate=tf.tanh,init_params_flag=init_params_flag)
					layer_mu=tf.reshape(layer_mu,[-1,n_steps,dim])
					params.append(layer_mu)
				with tf.variable_scope('vd_fc_cov') as scope:
					pre_activate=layers.fc_layer("vd_fc_cov",layer,
							dim_out, dim,
							wd_w,wd_bias,
							activate=None,init_params_flag=init_params_flag)
					layer_cov = tf.nn.softplus(pre_activate, name=scope.name)
					layer_cov=tf.reshape(layer_cov,[-1,n_steps,dim])
					params.append(layer_cov)
			else:
				raise Exception('[Error] unknown emmition type')
	return params


# x_t+1 = f (x_t)
def computeTransitionFuncFromNN(in_points,n_steps,init_params_flag=True,control_params=None):
	"""
	mu: points x dim
	"""
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	
	wd_bias=None
	wd_w=0.1
	hy_param=hy.get_hyperparameter()
	layer=tf.reshape(in_points,[-1,dim])
	with tf.name_scope('transition') as scope_parent:
		with tf.variable_scope('transition_var') as v_scope_parent:
			if hy_param["transition_internal_layers"]:
				# z -> layer
				layer,dim_out=build_nn(layer,dim_input=dim,n_steps=n_steps,
					hyparam_name="transition_internal_layers",name="tr",
					init_params_flag=init_params_flag,
					control_params=control_params
					)

			# layer -> layer_mean
			with tf.variable_scope('tr_fc_out') as scope:
				layer_logit=layers.fc_layer("tr_fc_out",layer,
						dim_out, dim,
						wd_w,wd_bias,
						activate=None,init_params_flag=init_params_flag)
			if control_params["state_type"]=="discrete":
				layer_z=tf.nn.softmax(layer_logit)
			elif control_params["state_type"]=="normal":
				layer_z=layer_logit
			else:
				raise Exception('[Error] unknown state type')
	return layer_z


def sampleTransitionFromDist(z_param,n_steps,init_state,init_params_flag=True,control_params=None):
	if control_params["state_type"]=="normal" :
		eps=control_params["placeholders"]["tr_eps"]
		q_zz=computeTransitionUKF(z_param[0],z_param[1],n_steps,mean_prior0=init_state[0],cov_prior0=init_state[1],init_params_flag=init_params_flag,control_params=control_params)
		z_s=sample_normal(q_zz,eps)
	else:
		raise Exception('[Error] not supported dynamics_type=function & state_type=%s'%(control_params["state_type"],))
	return z_s,q_zz

def sampleTransition(z,n_steps,init_state,init_params_flag=True,control_params=None):
	if control_params["state_type"]=="discrete":
		q_zz=computeTransition(z,n_steps,init_state,init_params_flag=init_params_flag,control_params=control_params)
		#z_s=q_zz[0]
		dist=tf.contrib.distributions.OneHotCategorical(probs=q_zz[0])
		z_s=tf.cast(dist.sample(),tf.float32)
	elif control_params["state_type"]=="normal":
		q_zz=computeTransition(z,n_steps,init_state,init_params_flag=init_params_flag,control_params=control_params)
		eps=control_params["placeholders"]["tr_eps"]
		z_s=sample_normal(q_zz,eps)
	else:
		raise Exception('[Error] unknown state type')
	return z_s,q_zz

def computeTransition(z,n_steps,init_state,init_params_flag=True,control_params=None):
	"""
	mu: bs x T x dim
	cov: bs x T x dim
	prior0: bs x 1 x dim
	"""
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	# add initial state and remove last state
	in_z=tf.reshape(z,[-1,n_steps,dim])
	in_z=tf.concat([init_state,in_z[:,:-1,:]],axis=1)

	out_param=computeTransitionDistWithNN(in_z,n_steps,init_params_flag,control_params=control_params)
	return out_param
	
def p_filter(x,z,epsilon,sample_size,proposal_sample_size,batch_size,control_params):
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	dim_emit=hy_param["dim_emit"]
	
	resample_size=sample_size
	if control_params["pfilter_type"]=="trained_dynamics":
		if control_params["dynamics_type"]=="function":
			#  x: (sample_size x batch_size) x dim_emit
			#  z: (sample_size x batch_size) x dim
			mu_trans=computeTransitionFunc(z,1,control_params=control_params)
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
		elif control_params["dynamics_type"]=="distribution":
			out_params=computeTransitionDistWithNN(z,1,init_params_flag=True,control_params=control_params)
			mu_trans=out_params[0]
			cov_trans=out_params[1]+0.01
			
			proposal_dist=tf.contrib.distributions.Normal(mu_trans[:,0,:],cov_trans[:,0,:])
			particles=proposal_dist.sample(proposal_sample_size)
		else:
			raise Exception('[Error] unknown dynamics type')
	elif control_params["pfilter_type"]=="zero_dynamics":
		proposal_dist=tf.contrib.distributions.Normal(z,1.0)
		particles=proposal_dist.sample(proposal_sample_size)
	else:
		raise Exception('[Error] unknown pfilter type')
		
	#  particles: proposal_sample_size x (sample_size x batch_size) x dim
	#particles_d=particles-mu_trans[:,:]
	#particles_w=particles_d**2/cov_trans[:,:]
	#  particles: (proposal_sample_size x sample_size x batch_size) x dim
	particles =tf.reshape(particles,[-1,dim])
	obs_params=computeEmission(particles,1,control_params=control_params)
	#  mu: (proposal_sample_size x sample_size)  x batch_size x emit_dim
	#  cov: (proposal_sample_size x sample_size) x batch_size x emit_dim
	mu=obs_params[0]
	cov=obs_params[1]
	mu =tf.reshape(mu,[-1,batch_size,dim_emit])
	cov = tf.clip_by_value(tf.reshape(cov,[-1,batch_size,dim_emit]),1.0e-10,2)
	#  x: batch_size x emit_dim
	d=mu-x[:,:]
	w=-tf.reduce_sum(d**2/cov,axis=2)
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
	# particles: (resample) x b x dim
	out=tf.gather_nd(particles,particle_ids)
	outputs={
		"sampled_pred_params":[mu,cov],
		"sampled_z":out}
	return outputs

def get_init_state(batch_size,dim):
	init_s=np.zeros((1,1,dim),dtype=np.float32)
	init_s[:,:,0]=1
	init_state=tf.tile(tf.constant(init_s,dtype=np.float32),(batch_size,1,1))
	return init_state

def get_init_dist(batch_size,dim):
	init_m=np.zeros((1,1,dim),dtype=np.float32)
	init_mu=tf.tile(tf.constant(init_m,dtype=np.float32),(batch_size,1,1))
	init_v=np.ones((1,1,dim),dtype=np.float32)
	init_var=tf.tile(tf.constant(init_v,dtype=np.float32),(batch_size,1,1))
	return [init_mu,init_var]

def inference(n_steps,control_params):
	if control_params["dynamics_type"]=="distribution" :
		return inference_by_sample(n_steps,control_params)
	elif control_params["dynamics_type"]=="function" :
		return inference_by_dist(n_steps,control_params)
	else:
		raise Exception('[Error] unknown dynamics type')
	
def inference_by_dist(n_steps,control_params):
	"""
	Returns:
		indference results
	"""
	# get input data
	placeholders=control_params["placeholders"]
	x=placeholders["x"]
	pot_points=placeholders["potential_points"]
	# get parameters
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	dim_emit=hy_param["dim_emit"]
	bs=tf.shape(x)[0]
	with tf.name_scope('inference') as scope_parent:
		#z_q: (bs x T) x dim
		z_s,z_params=sampleVariationalDist(x,n_steps,init_params_flag=True,control_params=control_params)
		init_state=get_init_dist(bs,dim)
		z_pred_s,z_pred_params=sampleTransitionFromDist(z_params,n_steps,init_state,
				init_params_flag=True,control_params=control_params)
		pot_loss=None
		# compute emission
		obs_params=computeEmission(z_s,n_steps,init_params_flag=True,control_params=control_params)
		obs_pred_params=computeEmission(z_pred_s,n_steps,init_params_flag=False,control_params=control_params)
	outputs={"z_s":z_s,
			"z_params":z_params,
			"z_pred_s":z_pred_s,
			"z_pred_params":z_pred_params,
			"obs_params":obs_params,
			"obs_pred_params":obs_pred_params,
			"potential_loss":pot_loss}
	return outputs


def inference_by_sample(n_steps,control_params):
	"""
	Returns:
		indference results
	"""
	# get input data
	placeholders=control_params["placeholders"]
	x=placeholders["x"]
	pot_points=placeholders["potential_points"]
	# get parameters
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	dim_emit=hy_param["dim_emit"]
	bs=tf.shape(x)[0]
	with tf.name_scope('inference') as scope_parent:
		#z_q: (bs x T) x dim
		z_s,z_params=sampleVariationalDist(x,n_steps,init_params_flag=True,control_params=control_params)
		init_state=get_init_state(bs,dim)
		z_pred_s,z_pred_params=sampleTransition(z_s,n_steps,init_state,
				init_params_flag=True,control_params=control_params)
		pot_loss=None
		# compute emission
		obs_params=computeEmission(z_s,n_steps,init_params_flag=True,control_params=control_params)
		obs_pred_params=computeEmission(z_pred_s,n_steps,init_params_flag=False,control_params=control_params)
	outputs={"z_s":z_s,
			"z_params":z_params,
			"z_pred_s":z_pred_s,
			"z_pred_params":z_pred_params,
			"obs_params":obs_params,
			"obs_pred_params":obs_pred_params,
			"potential_loss":pot_loss}
	return outputs



def computeNegCLL(x,outputs,mask,control_params):
	eps=1.0e-10
	max_var=2
	cov_p=outputs["obs_params"][1]+eps
	mu_p=outputs["obs_params"][0]

	negCLL=tf.log(2*np.pi)+tf.log(cov_p)+(x-mu_p)**2/tf.clip_by_value(cov_p,eps,max_var)
	negCLL=negCLL*0.5
	negCLL= negCLL*mask
	negCLL = tf.reduce_sum(negCLL,axis=2)
	negCLL = tf.reduce_sum(negCLL,axis=1)
	return negCLL

def kl_normal(mu1,var1,mu2,var2):
	return tf.log(var2)/2.0-tf.log(var1)/2.0+var1/(2.0*var2)+(mu1-mu2)**2/(2.0*var2)-1/2
def computeTemporalKL(x,outputs,length,control_params):
	"""
	z: bs x T x dim
	prior0: bs x 1 x dim
	"""
	eps=1.0e-10
	max_var=2.0
	if control_params["state_type"]=="discrete":
		mu_p=outputs["z_pred_params"][0]
		mu_q=outputs["z_params"][0]	
		kl_t= mu_q* (tf.log(mu_q+eps)-tf.log(mu_p+eps))
	elif control_params["state_type"]=="normal":
		eps=1.0e-10
		cov_p=outputs["z_pred_params"][1]+eps
		mu_p=outputs["z_pred_params"][0]
		cov_q=outputs["z_params"][1]+eps
		mu_q=outputs["z_params"][0]
		kl_t=tf.log(cov_p)-tf.log(cov_q)-1+(cov_q+(mu_p-mu_q)**2)/tf.clip_by_value(cov_p,eps,max_var)
	else:
		raise Exception('[Error] unknown state type')

	#masked_kl=tf.reduce_sum(kl_t,axis=2)*mask
	mask=tf.sequence_mask(length,maxlen=kl_t.shape[1],dtype=tf.float32)
	kl_t=tf.reduce_sum(kl_t,axis=2)
	kl_t=kl_t*mask

	kl_t=tf.reduce_sum(kl_t,axis=1)
	return kl_t



def loss(outputs,alpha=1,control_params=None):
	"""Add 
	Returns:
		Loss tensor of type float.
	"""
	# get input data
	placeholders=control_params["placeholders"]
	x=placeholders["x"]
	mask=placeholders["m"]
	length=placeholders["s"]
	# loss
	negCLL=computeNegCLL(x,outputs,mask,control_params)
	temporalKL=computeTemporalKL(x,outputs,length,control_params)
	cost_pot=tf.constant(0.0,dtype=np.float32)
	if outputs["potential_loss"] is not None:
		pot=outputs["potential_loss"]
		#sum_pot=tf.reduce_sum(pot*mask,axis=1)
		sum_pot=tf.reduce_sum(pot,axis=1)
		cost_pot=tf.reduce_mean(pot)
	mean_cost = tf.reduce_mean(negCLL+alpha*temporalKL+alpha*1.0*cost_pot, name='train_cost')
	#cost_mean = tf.reduce_mean((1-alpha)*negCLL+alpha*temporalKL+alpha*1.0*cost_pot, name='train_cost')
	tf.add_to_collection('losses', mean_cost)

	# The total loss is defined as the cross entropy loss plus all of the weight
	# decay terms (L2 loss).
	total_cost=tf.add_n(tf.get_collection('losses'), name='total_loss')
	costs=[tf.reduce_mean(negCLL),tf.reduce_mean(temporalKL),cost_pot]
	diff=None
	if "obs_params" in outputs:
		#diff=tf.reduce_mean((x-outputs["obs_pred"][0])**2/outputs["obs_pred"][1])
		diff=tf.reduce_mean((x-outputs["obs_params"][0])**2)
	return {"cost":total_cost,"mean_cost":mean_cost,"all_costs":costs,"error":diff}


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


def computeTransitionUKF(mu,cov,n_steps,mean_prior0=None,cov_prior0=None,init_params_flag=True,control_params=None):
	"""
	mu: bs x T x dim
	cov: bs x T x dim
	prior0: bs x 1 x dim
	"""
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
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

	out_points=computeTransitionFunc(in_points,n_steps,init_params_flag=init_params_flag,control_params=control_params)
	# N x bs x T x dim
	layer_mean=tf.reshape(out_points,[dim*2+1,-1,n_steps,dim])
	temp_sigma=layer_mean-layer_mean[0,:,:,:]
	layer_cov=tf.reduce_sum(temp_sigma**2,axis=0)

	if mean_prior0 is not None and cov_prior0 is not None:
		output_mu=tf.concat([mean_prior0,layer_mean[0,:,:-1,:]],axis=1)
		output_cov =tf.concat([cov_prior0 ,layer_cov[:,:-1,:]],axis=1)
		return output_mu,output_cov
	else:
		return layer_mean[0,:,:,:],layer_cov

def computePotentialLoss(mu_q,cov_q,pot_points,n_steps,control_params=None):
	pot_loss=None
	if hy_param["potential_enabled"]:
			if hy_param["potential_grad_transition_enabled"]==False:
				use_data_points=False
				if pot_points is None:
					use_data_points=True
				if use_data_points:
					## compute V(x(t+1))-V(x(t)) < 0 for stability
					mu_trans_1,cov_trans_1,params_tr=computeTransitionUKF(mu_q,cov_q,n_steps,None,None,params_tr,init_params_flag=False,control_params=control_params)
					#mu_q: bs x T x dim
					#mu_trans_1: bs x T x dim
					params_pot=None
					pot0=computePotential(mu_q,n_steps,control_params=control_params)
					pot1=computePotential(mu_trans_1,n_steps,init_params_flag=False,control_params=control_params)
					#pot: bs x T
					c=0.1
					pot=tf.nn.relu(pot1-pot0+c)
					pot_loss=tf.reshape(pot,[-1,n_steps])
				else:
					mu_trans_1,params_tr=computeTransitionFunc(pot_points,1,init_params_flag=False,control_params=control_params)
					params_pot=None
					pot0,params_pot=computePotential(pot_points,1,control_params=control_params)
					pot1,params_pot=computePotential(mu_trans_1,1,init_params_flag=False,control_params=control_params)
					#pot: bs x T
					c=0.1
					pot=tf.nn.relu(pot1-pot0+c)
					pot_loss=tf.reshape(pot,[-1,1])
	return pot_loss

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
					layer_mean=layers.fc_layer("pot_fc_mean",layer,
							dim_out, 1,
							wd_w,wd_bias,
							activate=tf.sigmoid,init_params_flag=init_params_flag)
			else:
				layer=z
				layer_mean=layers.fc_layer("pot_fc_mean",layer,
					dim, 1,
					wd_w,wd_bias,
					activate=tf.sigmoid,init_params_flag=init_params_flag)
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
		

