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
import joblib
import json
import argparse

import dmm_input
#from dmm_model import inference_by_sample, loss, p_filter, sampleVariationalDist
from dmm_model import inference, loss, p_filter, sampleVariationalDist
from dmm_model import fivo
from dmm_model import construct_placeholder, computeEmission, computeVariationalDist
import hyopt as hy
from attractor import field,potential,make_griddata_discrete,compute_discrete_transition_mat

# for profiler
from tensorflow.python.client import timeline

#FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
#tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")

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
			return obj.tolist() # or map(int, obj)
		return json.JSONEncoder.default(self, obj)


def get_default_config():
	config={}
	# data and network
	#config["dim"]=None
	config["dim"]=2
	# training
	config["epoch"] = 10
	config["patience"] = 5
	config["batch_size"] = 100
	config["alpha"] = 1.0
	config["learning_rate"] = 1.0e-2
	config["curriculum_alpha"]=False
	config["epoch_interval_save"] =10#100
	config["epoch_interval_print"] =10#100
	config["sampling_tau"]=10#0.1
	config["normal_max_var"]=5.0#1.0
	config["normal_min_var"]=1.0e-5
	config["zero_dynamics_var"]=1.0
	config["pfilter_sample_size"]=10
	config["pfilter_proposal_sample_size"]=1000
	config["pfilter_save_sample_num"]=100
	# dataset
	config["train_test_ratio"]=[0.8,0.2]
	config["data_train_npy"] = None
	config["mask_train_npy"] = None
	config["data_test_npy"] = None
	config["mask_test_npy"] = None
	# save/load model
	config["save_model_path"] = None
	config["load_model"] = None
	config["save_result_train"]=None
	config["save_result_test"]=None
	config["save_result_filter"]=None
	#config["state_type"]="discrete"
	config["state_type"]="normal"
	config["sampling_type"]="none"
	config["time_major"]=True
	config["steps_npy"]=None
	config["steps_test_npy"]=None
	config["sampling_type"]="normal"
	config["emission_type"]="normal"
	config["state_type"]="normal"
	config["dynamics_type"]="distribution"
	config["pfilter_type"]="trained_dynamics"
	config["potential_enabled"]=True,
	config["potential_grad_transition_enabled"]=True,
	config["potential_nn_enabled"]=False,
 
	# generate json
	#fp = open("config.json", "w")
	#json.dump(config, fp, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

	return config

def construct_feed(idx,data,placeholders,alpha,is_train=False):
	feed_dict={}
	num_potential_points=100
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	dim_emit=hy_param["dim_emit"]
	n_steps=hy_param["n_steps"]
	batch_size=len(idx)

	dropout_rate=0.0
	if is_train:
		if "dropout_rate" in hy_param:
			dropout_rate=hy_param["dropout_rate"]
		else:
			dropout_rate=0.5
	# 
	for key,ph in placeholders.items():
		if key == "x":
			feed_dict[ph]=data.x[idx,:,:]
		elif key == "m":
			feed_dict[ph]=data.m[idx,:,:]
		elif key == "s":
			feed_dict[ph]=data.s[idx]
		elif key == "alpha":
			feed_dict[ph]=alpha
		elif key == "vd_eps":
			#eps=np.zeros((batch_size,n_steps,dim))
			if hy_param["state_type"]=="discrete":
				eps=np.random.uniform(1.0e-10,1.0-1.0e-10,(batch_size,n_steps,dim))
				eps=-np.log(-np.log(eps))
			else:
				eps=np.random.standard_normal((batch_size,n_steps,dim))
			feed_dict[ph]=eps
		elif key == "tr_eps":
			#eps=np.zeros((batch_size,n_steps,dim))
			eps=np.random.standard_normal((batch_size,n_steps,dim))
			feed_dict[ph]=eps
		elif key == "potential_points":
			pts=np.random.standard_normal((num_potential_points,dim))
			feed_dict[ph]=pts
		elif key == "dropout_rate":
			feed_dict[ph]=dropout_rate
		elif key == "is_train":
			feed_dict[ph]=is_train
	return feed_dict


def print_variables():
	# print variables
	print('## emission variables')
	vars_em = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="emission_var")
	for v in vars_em:
		print(v.name)
	print('## variational dist. variables')
	vars_vd = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="variational_dist_var")
	for v in vars_vd:
		print(v.name)
	print('## transition variables')
	vars_tr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="transition_var")
	for v in vars_tr:
		print(v.name)
	print('## potential variables')
	vars_pot = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="potential_var")
	for v in vars_pot:
		print(v.name)
	return

def compute_alpha(config,i):
	alpha_max=config["alpha"]
	if config["curriculum_alpha"]:
		begin_tau=config["epoch"]*0.1
		end_tau=config["epoch"]*0.9
		tau=100.0
		if i < begin_tau:
			alpha=0.0
		elif i<end_tau:
			alpha=alpha_max*(1.0-np.exp(-(i-begin_tau)/tau))
		else:
			alpha=alpha_max
		return alpha
	return alpha_max

class EarlyStopping:
	def __init__(self,config, **kwargs):
		self.prev_validation_cost=None
		self.validation_count=0
		self.config=config
	def evaluate_validation(self,validation_cost,info):
		config=self.config
		if self.prev_validation_cost is not None and self.prev_validation_cost<validation_cost:
			self.validation_count+=1
			if config["patience"] >0 and self.validation_count>=config["patience"]:
				self.print_info(info)
				print("[stop] by validation")
				return True
		else:
			self.validation_count=0
		self.prev_validation_cost=validation_cost
		return False
	def print_info(self,info):
		config=self.config
		epoch=info["epoch"]
		training_cost=info["training_cost"]
		validation_cost=info["validation_cost"]
		training_error=info["training_error"]
		validation_error=info["validation_error"]
		training_all_costs=info["training_all_costs"]
		validation_all_costs=info["validation_all_costs"]
		alpha=info["alpha"]
		save_path=info["save_path"]
		if save_path is None:
			format_tuple=(epoch, training_cost, training_error,
				validation_cost,validation_error, self.validation_count)
			print("epoch %d, training cost %g (error=%g), validation cost %g (error=%g) (count=%d) "%format_tuple)
			print("[LOG] %d, %g,%g,%g,%g, %g, %g,%g,%g, %g,%g,%g"%(epoch, 
				training_cost,validation_cost,training_error,validation_error,alpha,
				training_all_costs[0],training_all_costs[1],training_all_costs[2],
				validation_all_costs[0],validation_all_costs[1],validation_all_costs[2]))
		else:
			format_tuple=(epoch, training_cost,training_error,
				validation_cost,validation_error,self.validation_count,save_path)
			print("epoch %d, training cost %g (error=%g), validation cost %g (error=%g) (count=%d) ([SAVE] %s) "%format_tuple)
			print("[LOG] %d, %g,%g,%g,%g, %g, %g,%g,%g, %g,%g,%g"%(epoch, 
				training_cost,validation_cost,training_error,validation_error,alpha,
				training_all_costs[0],training_all_costs[1],training_all_costs[2],
				validation_all_costs[0],validation_all_costs[1],validation_all_costs[2]))

def compute_cost(sess,placeholders,data,data_idx,output_cost,batch_size,alpha,is_train):
	# initialize costs
	cost=0.0
	error=0.0
	all_costs=np.zeros((3,),np.float32)
	# compute cost in data
	n_batch=int(np.ceil(data.num*1.0/batch_size))
	for j in range(n_batch):
		idx=data_idx[j*batch_size:(j+1)*batch_size]
		feed_dict=construct_feed(idx,data,placeholders,alpha,is_train=is_train)
		cost     +=np.array(sess.run(output_cost["cost"],feed_dict=feed_dict))
		all_costs+=np.array(sess.run(output_cost["all_costs"],feed_dict=feed_dict))
		error    +=np.array(sess.run(output_cost["error"],feed_dict=feed_dict))/n_batch
	data_info={
			"cost":cost,
			"error":error,
			"all_costs":all_costs,
			}
	return data_info

def compute_cost_train_valid(sess,placeholders,train_data,valid_data,train_idx,valid_idx,output_cost,batch_size,alpha):
	train_data_info=compute_cost(sess,placeholders,train_data,train_idx,output_cost,batch_size,alpha,is_train=True)
	valid_data_info=compute_cost(sess,placeholders,valid_data,valid_idx,output_cost,batch_size,alpha,is_train=False)
	all_info={}
	for k,v in train_data_info.items():
		all_info["training_"+k]=v
	for k,v in valid_data_info.items():
		all_info["validation_"+k]=v
	return all_info

def compute_result(sess,placeholders,data,data_idx,outputs,batch_size,alpha):
	results={}
	n_batch=int(np.ceil(data.num*1.0/batch_size))
	for j in range(n_batch):
		idx=data_idx[j*batch_size:(j+1)*batch_size]
		feed_dict=construct_feed(idx,data,placeholders,alpha)
		for k,v in outputs.items():
			if v is not None:
				res=sess.run(v,feed_dict=feed_dict)
				if k in ["z_s"]:
					if k in results:
						results[k]=np.concatenate([results[k],res],axis=0)
					else:
						results[k]=res
				elif k in ["obs_params","obs_pred_params", "z_params","z_pred_params"]:
					if k in results:
						for i in range(len(res)):
							results[k][i]=np.concatenate([results[k][i],res[i]],axis=0)
					else:
						if type(res)==tuple:
							results[k]=list(res)
						else:
							results[k]=res
	for k,v in results.items():
		if k in ["z_s"]:
			print(k,v.shape)
		elif k in ["obs_params","obs_pred_params", "z_params","z_pred_params"]:
			if len(v)==1:
				print(k,v[0].shape)
			else:
				print(k,v[0].shape,v[1].shape)
	return results


def get_dim(config,hy_param,data):
	dim_emit=data.dim
	if config["dim"] is None:
		dim=dim_emit
		config["dim"]=dim
	else:
		dim=config["dim"]
	hy_param["dim"]=dim
	hy_param["dim_emit"]=dim_emit
	return dim,dim_emit

def train(sess,config):
	hy_param=hy.get_hyperparameter()
	train_data,valid_data = dmm_input.load_data(config,with_shuffle=True,with_train_test=True)

	batch_size,n_batch=get_batch_size(config,hy_param,train_data)
	dim,dim_emit=get_dim(config,hy_param,train_data)
	n_steps=train_data.n_steps
	hy_param["n_steps"]=n_steps
	print("train_data_size:",train_data.num)
	print("batch_size     :",batch_size)
	print("n_steps        :",n_steps)
	print("dim_emit       :",dim_emit)
	
	placeholders=construct_placeholder(config)
	control_params={
			"config":config,
			"placeholders":placeholders,
			}
	# inference
	#outputs=inference_by_sample(n_steps,control_params=control_params)
	outputs=inference(n_steps,control_params=control_params)
	# cost
	output_cost=loss(outputs,placeholders["alpha"],control_params=control_params)
	# train_step
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_step = tf.train.AdamOptimizer(config["learning_rate"]).minimize(output_cost["cost"])
	print_variables()
	saver = tf.train.Saver()
	# initialize
	init = tf.global_variables_initializer()
	sess.run(init)

	train_idx=list(range(train_data.num))
	valid_idx=list(range(valid_data.num))
	## training
	validation_count=0
	prev_validation_cost=0
	alpha=None
	early_stopping=EarlyStopping(config)
	print("[LOG] epoch, cost,cost(valid.),error,error(valid.),alpha,cost(recons.),cost(temporal),cost(potential),cost(recons.,valid.),cost(temporal,valid),cost(potential,valid)")
	for i in range(config["epoch"]):
		np.random.shuffle(train_idx)
		alpha=compute_alpha(config,i)
		training_info=compute_cost_train_valid(sess,placeholders,
				train_data,valid_data,train_idx,valid_idx,
				output_cost,batch_size,alpha)
		# save
		save_path=None
		if i%config["epoch_interval_save"] == 0:
			save_path = saver.save(sess, config["save_model_path"]+"/model.%05d.ckpt"%(i))
		# early stopping
		training_info["epoch"]=i
		training_info["alpha"]=alpha
		training_info["save_path"]=save_path
		if i%config["epoch_interval_print"] == 0:
			early_stopping.print_info(training_info)
		if i%100:
			if early_stopping.evaluate_validation(training_info["validation_cost"],training_info):
				break

		# update
		n_batch=int(np.ceil(train_data.num*1.0/batch_size))
		for j in range(n_batch):
			idx=train_idx[j*batch_size:(j+1)*batch_size]
			feed_dict=construct_feed(idx,train_data,placeholders,alpha,is_train=True)
			train_step.run(feed_dict=feed_dict)

	training_info=compute_cost_train_valid(sess,placeholders,
			train_data,valid_data,train_idx,valid_idx,
			output_cost,batch_size,alpha)
	print("[RESULT] training cost %g, validation cost %g, training error %g, validation error %g"%(
		training_info["training_cost"],
		training_info["validation_cost"],
		training_info["training_error"],
		training_info["validation_error"]))
	hy_param["evaluation"]=training_info
	# save hyperparameter
	if config["save_model"] is not None and config["save_model"]!="":
		save_model_path=config["save_model"]
		save_path = saver.save(sess, save_model_path)
		print("[SAVE] %s"%(save_path))
	hy.save_hyperparameter()
	## save results
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


def infer(sess,config):
	hy_param=hy.get_hyperparameter()
	_,test_data = dmm_input.load_data(config,with_shuffle=False,with_train_test=False,
			test_flag=True)
	batch_size,n_batch=get_batch_size(config,hy_param,test_data)
	dim,dim_emit=get_dim(config,hy_param,test_data)
	n_steps=test_data.n_steps
	hy_param["n_steps"]=n_steps
	print("test_data_size:",test_data.num)
	print("batch_size     :",batch_size)
	print("n_steps        :",n_steps)
	print("dim_emit       :",dim_emit)
	alpha=config["alpha"]
	print("alpha          :",alpha)
	
	placeholders=construct_placeholder(config)
	control_params={
			"config":config,
			"placeholders":placeholders,
			}
	# inference
	outputs=inference(n_steps,control_params)
	# cost
	output_cost=loss(outputs,placeholders["alpha"],control_params=control_params)
	# train_step
	saver = tf.train.Saver()
	print_variables()
	print("[LOAD]",config["load_model"])
	saver.restore(sess,config["load_model"])
	test_idx=list(range(test_data.num))
	# check point
	test_info=compute_cost(sess,placeholders,
			test_data,test_idx,
			output_cost,batch_size,alpha,is_train=False)
	print("cost: %g"%(test_info["cost"]))

	## save results
	if config["save_result_test"]!="":
		results=compute_result(sess,placeholders,test_data,test_idx,outputs,batch_size,alpha)
		results["config"]=config
		print("[SAVE] result : ",config["save_result_test"])
		base_path = os.path.dirname(config["save_result_test"])
		os.makedirs(base_path,exist_ok=True)
		joblib.dump(results,config["save_result_test"])
	

def filter_discrete_forward(sess,config):
	hy_param=hy.get_hyperparameter()
	_,test_data = dmm_input.load_data(config,with_shuffle=False,with_train_test=False,test_flag=True)
	batch_size,n_batch=get_batch_size(config,hy_param,test_data)
	dim,dim_emit=get_dim(config,hy_param,test_data)
	
	n_steps=1
	hy_param["n_steps"]=n_steps
	z_holder=tf.placeholder(tf.float32,shape=(None,dim))
	z0=make_griddata_discrete(dim)
	batch_size=z0.shape[0]
	control_params={
		"dropout_rate":0.0,
		"config":config,
		}
	# inference
	params=computeEmission(z_holder,n_steps,
		init_params_flag=True,control_params=control_params)
	
	x_holder=tf.placeholder(tf.float32,shape=(None,100,dim_emit))
	qz=computeVariationalDist(x_holder,n_steps,init_params_flag=True,control_params=control_params)
	# load
	try:
		saver = tf.train.Saver()
		print("[LOAD] ",config["load_model"])
		saver.restore(sess,config["load_model"])
	except:
		print("[SKIP] Load parameters")
	#
	feed_dict={z_holder:z0}
	x_params=sess.run(params,feed_dict=feed_dict)
	
	x=test_data.x
	feed_dict={x_holder:x}
	out_qz=sess.run(qz,feed_dict=feed_dict)
	print(out_qz[0].shape)
	print(len(out_qz))
	# data:
	# data_num x n_steps x emit_dim
	num_d=x_params[0].shape[0]
	dist_x=[]
	
	for d in range(num_d):
		m=x_params[0][d,0,:]
		print("##,",d,",".join(map(str,m)))
	for d in range(num_d):
		cov=x_params[1][d,0,:]
		print("##,",d,",".join(map(str,cov)))
	for d in range(num_d):
		m=x_params[0][d,0,:]
		cov=x_params[1][d,0,:]
		diff_x=-(x-m)**2/(2*cov)
		prob=-1.0/2.0*np.log(2*np.pi*cov)+diff_x
		# prob: data_num x n_steps x emit_dim
		prob=np.mean(prob,axis=2)
		dist_x.append(prob)
	dist_x=np.array(dist_x)
	dist_x=np.transpose(dist_x,[1,2,0])
	# dist: data_num x n_steps x dim
	dist_x_max=np.zeros_like(dist_x)
	for i in range(dist_x.shape[0]):
		for j in range(dist_x.shape[1]):
			k=np.argmax(dist_x[i,j,:])
			dist_x_max[i,j,k]=1
	##
	## p(x|z)*q(z)
	## p(x,z)
	dist_qz=out_qz[0].reshape((20,100,dim))
	dist_pxz=dist_qz*np.exp(dist_x)
	##
	tr_mat=compute_discrete_transition_mat(sess,config)
	print(tr_mat)
	beta=5.0e-2
	tr_mat=beta*tr_mat+(1.0-beta)*np.identity(dim)
	print(tr_mat)
	## viterbi
	prob_viterbi=np.zeros_like(dist_x)
	prob_viterbi[:,:,:]=-np.inf
	path_viterbi=np.zeros_like(dist_x)
	index_viterbi=np.zeros_like(dist_x,dtype=np.int32)
	for d in range(dist_x.shape[0]):
		prob_viterbi[d,0,:]=dist_pxz[d,0,:]
		index_viterbi[d,0,:]=np.argmax(dist_pxz[d,0,:])
		step=dist_x.shape[1]-1
		for t in range(step):
			for i in range(dim):
				for j in range(dim):
					p=0
					p+=prob_viterbi[d,t,i]
					p+=np.log(dist_pxz[d,t+1,j])
					#p+=np.log(dist_qz[d,t+1,j])
					p+=np.log(tr_mat[i,j])
					"""
					if i==j:
						p+=np.log(tr_mat[i,j]*0.9)
					else:
						p+=np.log(tr_mat[i,j]*0.1)
					"""
					if prob_viterbi[d,t+1,j]<p:
						prob_viterbi[d,t+1,j]=p
						index_viterbi[d,t+1,j]=i
			##
		i=np.argmax(prob_viterbi[d,step,:])
		path_viterbi[d,step,i]=1.0
		for t in range(step):
			j=index_viterbi[d,step-t-1,i]
			#print(prob_viterbi[d,step-t-1,i])
			path_viterbi[d,step-t-1,j]=1.0
			i=j


	## save results
	if config["save_result_filter"]!="":
		results={}
		#results["dist"]=dist_x
		results["dist_max"]=dist_x_max
		results["dist_qz"]=dist_qz
		results["dist_pxz"]=dist_pxz
		results["dist_px"]=dist_x
		results["dist_viterbi"]=path_viterbi
		results["tr_mat"]=tr_mat
		print("[SAVE] result : ",config["save_result_filter"])
		joblib.dump(results,config["save_result_filter"])


def get_batch_size(config,hy_param,data):
	batch_size=config["batch_size"]
	n_batch=int(data.num/batch_size)
	if n_batch==0:
		batch_size=data.num
		n_batch=1
	elif n_batch*batch_size!=data.num:
		n_batch+=1
	return batch_size,n_batch

def construct_filter_placeholder(config):
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	dim_emit=hy_param["dim_emit"]
	n_steps=hy_param["n_steps"]
	# 
	x_holder=tf.placeholder(tf.float32,shape=(None,dim_emit))
	z_holder=tf.placeholder(tf.float32,shape=(None,dim))
	dropout_rate=tf.placeholder(tf.float32)
	is_train=tf.placeholder(tf.bool)
	#
	placeholders={"x":x_holder,
			"z":z_holder,
			"dropout_rate": dropout_rate,
			"is_train": is_train,
			}
	return placeholders

def construct_filter_feed(idx,batch_size,step,data,z,placeholders,is_train=False):
	feed_dict={}
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	dim_emit=hy_param["dim_emit"]
	n_steps=hy_param["n_steps"]

	dropout_rate=0.0
	if is_train:
		if "dropout_rate" in hy_param:
			dropout_rate=hy_param["dropout_rate"]
		else:
			dropout_rate=0.5
	# 
	for key,ph in placeholders.items():
		if key == "x":
			if idx+batch_size>data.num: # for last
				x=np.zeros((batch_size,dim),dtype=np.float32)
				bs=batch_size-(idx+batch_size-data.num)
				x[:bs,:]=data.x[idx:idx+batch_size,step,:]
			else:
				x=data.x[idx:idx+batch_size,step,:]
				bs=batch_size
			feed_dict[ph]=x
		elif key == "z":
			feed_dict[ph]=z
		elif key == "dropout_rate":
			feed_dict[ph]=dropout_rate
		elif key == "is_train":
			feed_dict[ph]=is_train
	return feed_dict,bs



def filtering(sess,config):
	hy_param=hy.get_hyperparameter()
	_,test_data = dmm_input.load_data(config,with_shuffle=False,with_train_test=False,test_flag=True)
	n_steps=test_data.n_steps
	hy_param["n_steps"]=n_steps
	dim,dim_emit=get_dim(config,hy_param,test_data)
	batch_size,n_batch=get_batch_size(config,hy_param,test_data)

	print("data_size",test_data.num,
		"batch_size",batch_size,
		", n_step",test_data.n_steps,
		", dim_emit",test_data.dim)
	placeholders=construct_filter_placeholder(config)

	sample_size=config["pfilter_sample_size"]
	proposal_sample_size=config["pfilter_proposal_sample_size"]
	save_sample_num=config["pfilter_save_sample_num"]
	#z0=np.zeros((batch_size*sample_size,dim),dtype=np.float32)
	z0=np.random.normal(0,1.0,size=(batch_size*sample_size,dim))
	
	control_params={
		"config":config,
		"placeholders":placeholders,
		}
	# inference
	#outputs=p_filter(x_holder,z_holder,None,dim,dim_emit,sample_size,batch_size,control_params=control_params)
	outputs=p_filter(placeholders["x"],placeholders["z"],None,sample_size,proposal_sample_size,batch_size,control_params=control_params)
	# loding model
	print_variables()
	saver = tf.train.Saver()
	print("[LOAD]",config["load_model"])
	saver.restore(sess,config["load_model"])
	
	feed_dict,_=construct_filter_feed(0,batch_size,0,test_data,z0,placeholders)
	result=sess.run(outputs,feed_dict=feed_dict)
	z=np.reshape(result["sampled_z"],[-1,dim])
	zs=np.zeros((sample_size,test_data.num,n_steps,dim),dtype=np.float32)
	
	# max: proposal_sample_size*sample_size
	sample_idx=list(range(proposal_sample_size*sample_size))
	np.random.shuffle(sample_idx)
	sample_idx=sample_idx[:save_sample_num]
	mus=np.zeros((save_sample_num,test_data.num,n_steps,dim_emit),dtype=np.float32)
	errors=np.zeros((save_sample_num,test_data.num,n_steps,dim_emit),dtype=np.float32)
	for j in range(n_batch):
		idx=j*batch_size
		print(j,"/",n_batch)
		for step in range(n_steps):
			feed_dict,bs=construct_filter_feed(idx,batch_size,step,test_data,z,placeholders)
			result=sess.run(outputs,feed_dict=feed_dict)
			z=result["sampled_z"]
			mu=result["sampled_pred_params"][0]
			zs[:,idx:idx+batch_size,step,:]=z[:,:bs,:]
			mus[:,idx:idx+batch_size,step,:]=mu[sample_idx,:bs,:]
			x=feed_dict[placeholders["x"]]
			errors[:,idx:idx+batch_size,step,:]=mu[sample_idx,:bs,:]-x[:bs,:]
			z=np.reshape(z,[-1,dim])
			print("*", end="")
		print("")
	## save results
	if config["save_result_filter"]!="":
		results={}
		results["z"]=zs
		results["mu"]=mus
		results["error"]=errors
		print("[SAVE] result : ",config["save_result_filter"])
		joblib.dump(results,config["save_result_filter"])

def construct_fivo_placeholder(config):
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	dim_emit=hy_param["dim_emit"]
	n_steps=hy_param["n_steps"]
	# 
	x_holder=tf.placeholder(tf.float32,shape=(None,n_steps,dim_emit))
	z0_holder=tf.placeholder(tf.float32,shape=(None,dim_emit))
	dropout_rate=tf.placeholder(tf.float32)
	is_train=tf.placeholder(tf.bool)
	#
	placeholders={"x":x_holder,
			"z":z0_holder,
			"dropout_rate": dropout_rate,
			"is_train": is_train,
			}
	return placeholders


def construct_fivo_feed(data_idx,batch_size,step,data,placeholders,is_train=False):
	feed_dict={}
	hy_param=hy.get_hyperparameter()
	dim=hy_param["dim"]
	dim_emit=hy_param["dim_emit"]
	n_steps=hy_param["n_steps"]
	sample_size=hy_param["pfilter_sample_size"]

	dropout_rate=0.0
	if is_train:
		if "dropout_rate" in hy_param:
			dropout_rate=hy_param["dropout_rate"]
		else:
			dropout_rate=0.5
	# 
	for key,ph in placeholders.items():
		if key == "x":
			idx=data_idx[step*batch_size:(step+1)*batch_size]
			if len(idx)<batch_size:
				x=np.zeros((batch_size,n_steps,dim),dtype=np.float32)
				x[:len(idx),:,:]=data.x[idx,:,:]
			else:
				x=data.x[idx,:,:]
			feed_dict[ph]=x
		elif key == "z":
			z0=np.random.normal(0,1.0,size=(batch_size*sample_size,dim_emit))
			feed_dict[ph]=z0
		elif key == "dropout_rate":
			feed_dict[ph]=dropout_rate
		elif key == "is_train":
			feed_dict[ph]=is_train
	return feed_dict



def train_fivo(sess,config):
	hy_param=hy.get_hyperparameter()
	#_,test_data = dmm_input.load_data(config,with_shuffle=False,with_train_test=False,test_flag=True)
	train_data,valid_data = dmm_input.load_data(config,with_shuffle=True,with_train_test=True)
	n_steps=train_data.n_steps
	hy_param["n_steps"]=n_steps
	dim,dim_emit=get_dim(config,hy_param,train_data)
	batch_size,n_batch=get_batch_size(config,hy_param,train_data)

	print("data_size",train_data.num,
		"batch_size",batch_size,
		", n_step",train_data.n_steps,
		", dim_emit",train_data.dim)
	placeholders=construct_fivo_placeholder(config)

	sample_size=config["pfilter_sample_size"]
	proposal_sample_size=config["pfilter_proposal_sample_size"]
	save_sample_num=config["pfilter_save_sample_num"]
	#z0=np.zeros((batch_size*sample_size,dim),dtype=np.float32)
	z0=np.random.normal(0,1.0,size=(batch_size*sample_size,dim))
	control_params={
		"config":config,
		"placeholders":placeholders,
		}
	# inference
	#outputs=p_filter(x_holder,z_holder,None,dim,dim_emit,sample_size,batch_size,control_params=control_params)
	output_cost=fivo(placeholders["x"],placeholders["z"],None,n_steps,sample_size,proposal_sample_size,batch_size,control_params=control_params)
	##========##
	# train_step
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_step = tf.train.AdamOptimizer(config["learning_rate"]).minimize(output_cost["cost"])
	print_variables()
	saver = tf.train.Saver()
	if config["profile"]:
		vars_to_train = tf.trainable_variables()
		print(vars_to_train)
		writer = tf.summary.FileWriter('logs', sess.graph)
	# initialize
	init = tf.global_variables_initializer()
	sess.run(init)

	train_idx=list(range(train_data.num))
	valid_idx=list(range(valid_data.num))
	## training
	validation_count=0
	prev_validation_cost=0
	alpha=None
	early_stopping=EarlyStopping(config)
	print("[LOG] epoch, cost,cost(valid.),error,error(valid.),alpha,cost(recons.),cost(temporal),cost(potential),cost(recons.,valid.),cost(temporal,valid),cost(potential,valid)")
	for i in range(config["epoch"]):
		np.random.shuffle(train_idx)
		alpha=compute_alpha(config,i)
		
		# save
		save_path=None
		if i%config["epoch_interval_save"] == 0:
			save_path = saver.save(sess, config["save_model_path"]+"/model.%05d.ckpt"%(i))
		# early stopping
		
		# update
		n_batch=int(np.ceil(train_data.num*1.0/batch_size))
		profiler_start=False
		cost=0
		for j in range(n_batch):
			print(j,"/",n_batch)
			run_metadata=None
			run_options=None
			if config["profile"] and j==1 and i==2:
				profiler_start=True
				run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
				run_metadata = tf.RunMetadata()
			feed_dict=construct_fivo_feed(train_idx,batch_size,j,train_data,placeholders,is_train=True)
			train_step.run(feed_dict=feed_dict)
			c=sess.run(output_cost["cost"],feed_dict=feed_dict)
			cost+=c
			if profiler_start:
				step_stats = run_metadata.step_stats
				tl = timeline.Timeline(step_stats)
				ctf = tl.generate_chrome_trace_format(
					show_memory=False,
					show_dataflow=True)
				with open("logs/timeline.json", "w") as f:
					f.write(ctf)
				print("[SAVE] logs/timeline.json")
				profiler_start=False


		print(cost/n_batch)
		###
		###
		#result=sess.run(outputs,feed_dict=feed_dict)
	# save hyperparameter
	if config["save_model"] is not None and config["save_model"]!="":
		save_model_path=config["save_model"]
		save_path = saver.save(sess, save_model_path)
		print("[SAVE] %s"%(save_path))
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
	print("[LOAD]",config["load_model"])
	saver.restore(sess,config["load_model"])
	
	feed_dict={x_holder:test_data.x[0:batch_size,:,:],z0_holder:z0}
	result=sess.run(outputs,feed_dict=feed_dict)

	z=np.reshape(result["sampled_z"],[-1,dim])
	zs=np.zeros((sample_size,test_data.num,n_steps,dim),dtype=np.float32)
	
	# max: proposal_sample_size*sample_size
	sample_idx=list(range(proposal_sample_size*sample_size))
	np.random.shuffle(sample_idx)
	sample_idx=sample_idx[:save_sample_num]
	mus=np.zeros((sample_size,test_data.num,n_steps,dim_emit),dtype=np.float32)
	errors=np.zeros((sample_size,test_data.num,n_steps,dim_emit),dtype=np.float32)
	for j in range(n_batch):
		idx=j*batch_size
		print(j,"/",n_batch)
		if idx+batch_size>test_data.num: # for last
			x=np.zeros((batch_size,n_steps,dim),dtype=np.float32)
			bs=batch_size-(idx+batch_size-test_data.num)
			x[:bs,:,:]=test_data.x[idx:idx+batch_size,:,:]
		else:
			x=test_data.x[idx:idx+batch_size,:,:]
			bs=batch_size
		feed_dict={x_holder:x,z0_holder:z}
		
		###
		###
		result=sess.run(outputs,feed_dict=feed_dict)
		z=result["sampled_z"]
		obs_list=result["sampled_pred_params"]
		###
		###
		for step in range(n_steps):
			mu=obs_list[step][0]
			zs[:,idx:idx+batch_size,step,:]=z[step][:,:bs,:]
			print("======")
			#mus:save_sample_num,test_data.num,n_steps,dim_emit
			print(mus.shape)
			print(mu.shape)
			print("======")
			mus[:,idx:idx+batch_size,step,:]=mu[:,:bs,:]
			errors[:,idx:idx+batch_size,step,:]=mu[:,:bs,:]-x[:bs,step,:]
		z=np.reshape(z,[-1,dim])
		print("*", end="")
		print("")
		##
	
	## save results
	if config["save_result_filter"]!="":
		results={}
		results["z"]=zs
		results["mu"]=mus
		results["error"]=errors
		print("[SAVE] result : ",config["save_result_filter"])
		joblib.dump(results,config["save_result_filter"])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('mode', type=str,
			help='train/infer')
	parser.add_argument('--config', type=str,
			default=None,
			nargs='?',
			help='config json file')
	parser.add_argument('--no-config',
			action='store_true',
			help='use default setting')
	parser.add_argument('--save-config',
			default=None,
			nargs='?',
			help='save config json file')
	parser.add_argument('--model', type=str,
			default=None,
			help='model')
	parser.add_argument('--hyperparam', type=str,
			default=None,
			nargs='?',
			help='hyperparameter json file')
	parser.add_argument('--cpu',
			action='store_true',
			help='cpu mode (calcuration only with cpu)')
	parser.add_argument('--gpu', type=str,
			default=None,
			help='constraint gpus (default: all) (e.g. --gpu 0,2)')
	parser.add_argument('--profile',
			action='store_true',
			help='')
	args=parser.parse_args()
	# config
	config=get_default_config()
	if args.config is None:
		if not args.no_config:
			parser.print_help()
			#quit()
	else:
		fp = open(args.config, 'r')
		config.update(json.load(fp))
	#if args.hyperparam is not None:
	hy.initialize_hyperparameter(args.hyperparam)
	config.update(hy.get_hyperparameter())
	hy.get_hyperparameter().update(config)
	# gpu/cpu
	if args.cpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = ""
	elif args.gpu is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	# profile
	config["profile"]=args.profile
	# setup
	mode_list=args.mode.split(",")
	#with tf.Graph().as_default(), tf.device('/cpu:0'):
	for mode in mode_list:
		with tf.Graph().as_default():
			with tf.Session() as sess:
				# mode
				if mode=="train":
					train(sess,config)
				elif mode=="infer" or mode=="test":
					if args.model is not None:
						config["load_model"]=args.model
					infer(sess,config)
				elif mode=="filter":
					if args.model is not None:
						config["load_model"]=args.model
					filtering(sess,config)
				elif mode=="filter2":
					filter_discrete_forward(sess,config)
				elif mode=="train_fivo":
					train_fivo(sess,config)
				elif mode=="field":
					field(sess,config)
				elif mode=="potential":
					potential(sess,config)
	if args.save_config is not None:
		print("[SAVE] config: ",args.save_config)
		fp = open(args.save_config, "w")
		json.dump(config, fp, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '),cls=NumPyArangeEncoder)


