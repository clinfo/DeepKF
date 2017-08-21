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

import dkf_input
from dkf_model import inference, loss, p_filter
import hyopt as hy

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")


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
	# dataset
	config["train_test_ratio"]=[0.8,0.2]
	config["data_train_npy"] = "data/pack_data_emit.npy"
	config["mask_train_npy"] = "data/pack_mask_emit.npy"
	config["data_test_npy"] = "data/pack_data_emit_test.npy"
	config["mask_test_npy"] = "data/pack_mask_emit_test.npy"
	# save/load model
	config["save_model_path"] = "./model/"
	config["load_model"] = "./model/model.last.ckpt"
	config["save_result_train"]="./result/train.jbl"
	config["save_result_test"]="./result/test.jbl"
	config["save_result_filter"]="./result/filter.jbl"
	# generate json
	#fp = open("config.json", "w")
	#json.dump(config, fp, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

	return config


def train(sess,config):
	hy_param=hy.get_hyperparameter()
	x, m,x_val,m_val = dkf_input.inputs(config,with_shuffle=True,with_train_test=True)
	batch_size=config["batch_size"]
	n_batch=int(x.shape[0]/batch_size)
	n_steps=x.shape[1]
	if n_batch==0:
		batch_size=x.shape[0]
		n_batch=1
	dim_emit=x.shape[2]
	if config["dim"] is None:
		dim=dim_emit
		config["dim"]=dim
	else:
		dim=config["dim"]
	print("data_size",x.shape[0],"batch_size",batch_size,", n_step",x.shape[1],", dim_emit",x.shape[2])
	x_holder=tf.placeholder(tf.float32,shape=(None,n_steps,dim_emit))
	m_holder=tf.placeholder(tf.float32,shape=(None,n_steps))
	eps_holder=tf.placeholder(tf.float32,shape=(None,dim))
	potential_points_holder=tf.placeholder(tf.float32,shape=(None,dim))
	num_potential_points=100
	alpha_holder=tf.placeholder(tf.float32)
	control_params={"dropout_rate":0.5}
	
	# inference
	outputs=inference(x_holder,eps_holder,n_steps,dim,dim_emit,pot_points=potential_points_holder,control_params=control_params)
	# cost
	total_cost,cost_mean,costs=loss(x_holder,outputs,m_holder,alpha_holder,control_params=control_params)
	diff=tf.reduce_mean((x_holder-outputs["pred_params"][0])**2)
	# train_step
	train_step = tf.train.AdamOptimizer(1e-3).minimize(total_cost)
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
	# initialize
	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	sess.run(init)

	data_idx=list(range(x.shape[0]))
	## training
	validation_count=0
	prev_validation_cost=0
	alpha=config["alpha"]
	alpha_mode=True
	alpha=0.0
	for i in range(config["epoch"]):
		np.random.shuffle(data_idx)
		if alpha_mode:
			begin_tau=config["epoch"]*0.1
			end_tau=config["epoch"]*0.9
			tau=100.0
			if i < begin_tau:
				alpha=0.0
			elif i<end_tau:
				alpha=1.0-np.exp(-(i-begin_tau)/tau)
			else:
				alpha=1.0	
		if i%10 == 0:
			# check point
			cost=0.0
			validation_cost=0.0
			error=0.0
			all_costs=np.zeros((3,),np.float32)
			validation_all_costs=np.zeros((3,),np.float32)
			# compute cost in training data
			for j in range(n_batch):
				idx=data_idx[j*batch_size:(j+1)*batch_size]
				eps=np.zeros((batch_size*n_steps,dim))
				feed_dict={x_holder:x[idx,:,:],m_holder:m[idx,:],eps_holder:eps,alpha_holder:alpha}
				if potential_points_holder is not None:
					pts=np.random.standard_normal((num_potential_points,dim))
					feed_dict[potential_points_holder]=pts
				cost+=total_cost.eval(feed_dict=feed_dict)
				all_costs+=np.array(sess.run(costs,feed_dict=feed_dict))
				error+=diff.eval(feed_dict=feed_dict)/n_batch
			# compute cost in validation data
			eps=np.zeros((x_val.shape[0]*x_val.shape[1],dim))
			feed_dict={x_holder:x_val,m_holder:m_val,eps_holder:eps,alpha_holder:alpha}
			if potential_points_holder is not None:
				pts=np.random.standard_normal((num_potential_points,dim))
				feed_dict[potential_points_holder]=pts
			validation_cost+=total_cost.eval(feed_dict=feed_dict)
			validation_all_costs+=np.array(sess.run(costs,feed_dict=feed_dict))
			# save
			save_path = saver.save(sess, config["save_model_path"]+"/model.%05d.ckpt"%(i))
			# early stopping
			if prev_validation_cost<validation_cost:
				validation_count+=1
				if config["patience"] >0 and validation_count>=config["patience"]:
					print("step %d, training cost %g, validation cost %g (%s)[alpha=%g]"%(i, cost, validation_cost,save_path,alpha))
					print("[stop] by validation")
					break;
			else:
				validation_count=0
			print("step %d, training cost %g, validation cost %g (%s) [error=%g,alpha=%g]"%(i, cost, validation_cost,save_path,error,alpha))
			print("  training:[%g,%g,%g] validation:[%g,%g,%g]"%(all_costs[0],all_costs[1],all_costs[2],validation_all_costs[0],validation_all_costs[1],validation_all_costs[2]))
			print("[LOG] %d, %g,%g,%g,%g, %g,%g,%g, %g,%g,%g"%(i, cost,validation_cost,error,alpha, all_costs[0],all_costs[1],all_costs[2],validation_all_costs[0],validation_all_costs[1],validation_all_costs[2]))
			prev_validation_cost=validation_cost
		# update
		for j in range(n_batch):
			eps=np.random.standard_normal((batch_size*n_steps,dim))
			idx=data_idx[j*batch_size:(j+1)*batch_size]
			feed_dict={x_holder:x[idx,:,:],m_holder:m[idx,:],eps_holder:eps,alpha_holder:alpha}
			if potential_points_holder is not None:
				pts=np.random.standard_normal((num_potential_points,dim))
				feed_dict[potential_points_holder]=pts
			train_step.run(feed_dict=feed_dict)

	print("[RESULT] training cost %g, validation cost %g, error %g"%(cost, validation_cost,error))
	print("  training:[%g,%g,%g] validation:[%g,%g,%g]"%(all_costs[0],all_costs[1],all_costs[2],validation_all_costs[0],validation_all_costs[1],validation_all_costs[2]))
	output={"cost":cost,"validation_cost": validation_cost, "error":error,"all_costs":all_costs,"validation_all_costs":validation_all_costs}
	hy_param["evaluation"]=output
	# save hyperparameter
	save_model_path=config["save_model_path"]+"/model.last.ckpt"
	hy_param["load_model"]=save_model_path
	hy_param["save_model"]=""
	hy.save_hyperparameter()
	## save results
	save_path = saver.save(sess, save_model_path)
	print("[SAVE] %s"%(save_path))
	if config["save_result_train"]!="":
		results={}
		eps=np.zeros((x.shape[0]*x.shape[1],dim))
		for k,v in outputs.items():
			if v is not None:
				feed_dict={x_holder:x,m_holder:m,eps_holder:eps,alpha_holder:alpha}
				res=sess.run(v,feed_dict=feed_dict)
				results[k]=res
		results["x"]=x
		results["config"]=config
		joblib.dump(results,config["save_result_train"])


def infer(sess,config):
	_,_,x, m = dkf_input.inputs(config,with_shuffle=False,with_train_test=False,test_flag=True)
	batch_size=config["batch_size"]
	n_batch=int(x.shape[0]/batch_size)
	n_steps=x.shape[1]
	if n_batch==0:
		batch_size=x.shape[0]
		n_batch=1
	elif n_batch*batch_size!=x.shape[0]:
		n_batch+=1
	dim_emit=x.shape[2]
	if config["dim"] is None:
		dim=dim_emit
		config["dim"]=dim
	else:
		dim=config["dim"]
	print("data_size",x.shape[0],"batch_size",batch_size,", n_step",x.shape[1],", dim_emit",x.shape[2])
	x_holder=tf.placeholder(tf.float32,shape=(None,n_steps,dim_emit))
	m_holder=tf.placeholder(tf.float32,shape=(None,n_steps))
	control_params={"dropout_rate":0.0}
	
	# inference
	outputs=inference(x_holder,None,n_steps,dim,dim_emit,control_params=control_params)
	# cost
	total_cost,cost_mean,costs=loss(x_holder,outputs,m_holder,control_params=control_params)
	# train_step
	#init = tf.global_variables_initializer()
	#sess.run(init)
	saver = tf.train.Saver()# これ以前の変数のみ保存される
	print("[LOAD]",config["load_model"])
	saver.restore(sess,config["load_model"])
	data_idx=list(range(x.shape[0]))
	# check point
	cost=0.0
	for j in range(n_batch):
		idx=data_idx[j*batch_size:(j+1)*batch_size]
		feed_dict={x_holder:x[idx,:,:],m_holder:m[idx,:]}
		cost+=total_cost.eval(feed_dict=feed_dict)
	print("cost: %g"%(cost))

	## save results
	if config["save_result_test"]!="":
		results={}
		for k,v in outputs.items():
			if v is not None:
				feed_dict={x_holder:x,m_holder:m}
				res=sess.run(v,feed_dict=feed_dict)
				results[k]=res
		results["x"]=x
		print("[SAVE] result : ",config["save_result_test"])
		joblib.dump(results,config["save_result_test"])


def filtering(sess,config):
	_,_,x, m = dkf_input.inputs(config,with_shuffle=False,with_train_test=False,test_flag=True)
	batch_size=config["batch_size"]
	batch_size=10
	n_batch=int(x.shape[0]/batch_size)
	n_steps=x.shape[1]
	if n_batch==0:
		batch_size=x.shape[0]
		n_batch=1
	elif n_batch*batch_size!=x.shape[0]:
		n_batch+=1
	dim_emit=x.shape[2]
	if config["dim"] is None:
		dim=dim_emit
		config["dim"]=dim
	else:
		dim=config["dim"]
	print("data_size",x.shape[0],"batch_size",batch_size,", n_step",x.shape[1],", dim_emit",x.shape[2])
	x_holder=tf.placeholder(tf.float32,shape=(None,n_steps,dim_emit))
	m_holder=tf.placeholder(tf.float32,shape=(None,n_steps))
	z_holder=tf.placeholder(tf.float32,shape=(None,dim))
	step_holder=tf.placeholder(tf.int32)
	sample_size=10
	z0=np.zeros((batch_size*sample_size,dim),dtype=np.float32)
	control_params={"dropout_rate":0.0}
	# inference
	outputs=p_filter(x_holder,z_holder,step_holder,None,n_steps,dim,dim_emit,batch_size,control_params=control_params)
	# loding model
	saver = tf.train.Saver()# これ以前の変数のみ保存される
	print("[LOAD]",config["load_model"])
	saver.restore(sess,config["load_model"])
	
	feed_dict={x_holder:x[0:batch_size,:,:],z_holder:z0,step_holder:0}
	result=sess.run(outputs,feed_dict=feed_dict)
	print(result["sampled_z"].shape)
	print(result["sampled_pred_params"][0].shape)
	z=np.reshape(result["sampled_z"],[-1,dim])
	zs=np.zeros((sample_size,x.shape[0],n_steps,dim),dtype=np.float32)
	mus=np.zeros((sample_size*sample_size,x.shape[0],n_steps,dim_emit),dtype=np.float32)
	for j in range(n_batch):
		idx=j*batch_size
		print(j,"/",n_batch)
		for step in range(n_steps):
			feed_dict={x_holder:x[idx:idx+batch_size,:,:],z_holder:z,step_holder:step}
			bs=batch_size
			if idx+batch_size>x.shape[0]: # for last
				x2=np.zeros((batch_size,x.shape[1],x.shape[2]),dtype=np.float32)
				bs=batch_size-(idx+batch_size-x.shape[0])
				x2[:bs,:,:]=x[idx:idx+batch_size,:,:]
				feed_dict={x_holder:x2,z_holder:z,step_holder:step}
			result=sess.run(outputs,feed_dict=feed_dict)
			z=result["sampled_z"]
			mu=result["sampled_pred_params"][0]
			zs[:,idx:idx+batch_size,step,:]=z[:,:bs,:]
			mus[:,idx:idx+batch_size,step,:]=mu[:,:bs,:]
			z=np.reshape(z,[-1,dim])
	## save results
	if config["save_result_filter"]!="":
		results={}
		results["z"]=zs
		results["mu"]=mus
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
	# setup
	with tf.Graph().as_default():
	#with tf.Graph().as_default(), tf.device('/cpu:0'):
		with tf.Session() as sess:
			# mode
			if args.mode=="train":
				train(sess,config)
			elif args.mode=="infer":
				if args.model is not None:
					config["load_model"]=args.model
				infer(sess,config)
			elif args.mode=="filter":
				if args.model is not None:
					config["load_model"]=args.model
				filtering(sess,config)
	
	if args.save_config is not None:
		print("[SAVE] config: ",args.save_config)
		fp = open(args.save_config, "w")
		json.dump(config, fp, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '),cls=NumPyArangeEncoder)


