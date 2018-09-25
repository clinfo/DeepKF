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
from dmm_model import computeTransitionFunc,computeTransitionDistWithNN
from dmm_model import computeEmission, computePotential
import hyopt as hy

#FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")

def make_griddata(dim,nx,rx=1,max_dim=-1):
	arr=[]
	for d in range(dim):
		x = np.linspace(-rx, rx, nx)
		arr.append(x)
	grid = np.array(np.meshgrid(*arr)).reshape(dim,-1)
	grid=np.transpose(grid)
	if max_dim>0:
		new_grid=np.zeros((grid.shape[0],max_dim),np.float32)
		new_grid[:,:dim]=grid
		return new_grid
	return grid

def get_dim_emit(config):
	filename=config["data_train_npy"]
	x=np.load(filename)
	return x.shape[1]

def make_griddata_discrete(dim):
	return np.identity(dim, dtype=float)
	

def compute_discrete_transition_mat(sess,config):
	batch_size=config["batch_size"]
	dim_emit=get_dim_emit(config)
	if config["dim"] is None:
		dim=dim_emit
		config["dim"]=dim
	else:
		dim=config["dim"]
	n_steps=1
	z_holder=tf.placeholder(tf.float32,shape=(None,dim))
	z0=make_griddata_discrete(dim)
	batch_size=z0.shape[0]
	control_params={
		"dropout_rate":0.0,
		"config":config,
		"state_type":config["state_type"],
		"sampling_type":config["sampling_type"],
		"emission_type":config["emission_type"],
		"dynamics_type":config["dynamics_type"],
		}
	# inference
	#z_m,_,_=computeTransition(z_holder,n_steps,dim,mean_prior0=None,cov_prior0=None,params=None,without_cov=True)
	if control_params["dynamics_type"]=="distribution" :
		params=computeTransitionDistWithNN(z_holder,n_steps,control_params=control_params)
		z_m=params[0]
	elif control_params["dynamics_type"]=="function" :
		z_m = computeTransitionFunc(z_holder,n_steps,control_params=control_params)
	else:
		raise Exception('[Error] unknown dynamics type')
	# load
	try:
		saver = tf.train.Saver()
		print("[LOAD] ",config["load_model"])
		saver.restore(sess,config["load_model"])
	except:
		print("[SKIP] Load parameters")
	#
	feed_dict={z_holder:z0}
	z_next=sess.run(z_m,feed_dict=feed_dict)
	z_next=z_next.reshape((dim,dim))
	return z_next
	
def field_discrete(sess,config):
	z_next=compute_discrete_transition_mat(sess,config)
	#
	## save results
	"""
	if config["simulation_path"]!="":
		sim_filename=config["simulation_path"]+"/field.jbl"
		results={}
		results["z"]=z0
		results["z_next"]=z_next
		print("[SAVE] result : ",sim_filename)
		joblib.dump(results,sim_filename)
	"""

def field(sess,config):
	if config["state_type"]=="discrete" or config["state_type"]=="discrete_tr":
		return field_discrete(sess,config)
	else:
		return field_continuous(sess,config)

def field_continuous(sess,config):
	batch_size=config["batch_size"]
	dim_emit=get_dim_emit(config)
	if config["dim"] is None:
		dim=dim_emit
		config["dim"]=dim
	else:
		dim=config["dim"]
	n_steps=1
	z_holder=tf.placeholder(tf.float32,shape=(None,dim))
	z0=make_griddata(2,max_dim=dim,nx=30,rx=2.0)
	batch_size=z0.shape[0]
	control_params={
		"dropout_rate":0.0,
		"config":config,
		"state_type":config["state_type"],
		"sampling_type":config["sampling_type"],
		"emission_type":config["emission_type"],
		"dynamics_type":config["dynamics_type"],
		}
	# inference
	#z_m,_,_=computeTransition(z_holder,n_steps,dim,mean_prior0=None,cov_prior0=None,params=None,without_cov=True)
	if control_params["dynamics_type"]=="distribution" :
		params=computeTransitionDistWithNN(z_holder,n_steps,control_params=control_params)
		z_m=params[0]
	elif control_params["dynamics_type"]=="function" :
		z_m = computeTransitionFunc(z_holder,n_steps,control_params=control_params)
	else:
		raise Exception('[Error] unknown dynamics type')
	
	z_m =tf.reshape(z_m,[-1,dim])
	# grad
	g_z = tf.gradients(z_m, [z_holder])
	# load
	try:
		saver = tf.train.Saver()
		print("[LOAD] ",config["load_model"])
		saver.restore(sess,config["load_model"])
	except:
		print("[SKIP] Load parameters")
	#
	feed_dict={z_holder:z0}
	g=sess.run(g_z,feed_dict=feed_dict)
	#
	## save results
	if config["simulation_path"]!="":
		sim_filename=config["simulation_path"]+"/field.jbl"
		results={}
		results["z"]=z0
		results["gz"]=g
		print("[SAVE] result : ",sim_filename)
		joblib.dump(results,sim_filename)

def potential(sess,config):
	batch_size=config["batch_size"]
	dim_emit=get_dim_emit(config)
	if config["dim"] is None:
		dim=dim_emit
		config["dim"]=dim
	else:
		dim=config["dim"]
	n_steps=1
	z_holder=tf.placeholder(tf.float32,shape=(None,dim))
	z0=make_griddata(2,max_dim=dim,nx=30,rx=2.0)
	batch_size=z0.shape[0]
	control_params={"dropout_rate":0.0}
	# inference
	#z_m,_,_=computeTransition(z_holder,n_steps,dim,mean_prior0=None,cov_prior0=None,params=None,without_cov=True)
	z_m,_=computePotential(z_holder,n_steps,dim,params=None,control_params=control_params)
	
	# load
	saver = tf.train.Saver()# これ以前の変数のみ保存される
	print("[LOAD] ",config["load_model"])
	saver.restore(sess,config["load_model"])
	#
	feed_dict={z_holder:z0}
	g=sess.run(z_m,feed_dict=feed_dict)
	print(g)
	#
	## save results
	print(z0.shape)
	print(g.shape)
	if config["simulation_path"]!="":
		sim_filename=config["simulation_path"]+"/potential.jbl"
		results={}
		results["z"]=z0
		results["pot"]=g
		print("[SAVE] result : ",sim_filename)
		joblib.dump(results,sim_filename)



def infer(sess,config):
	batch_size=config["batch_size"]
	dim_emit=get_dim_emit(config)
	if config["dim"] is None:
		dim=dim_emit
		config["dim"]=dim
	else:
		dim=config["dim"]
	n_steps=1
	control_params={"dropout_rate":0.0}
	z_holder=tf.placeholder(tf.float32,shape=(None,dim))
	# inference
	z_m,_=computeTransitionFunc(z_holder,n_steps,dim,params=None,control_params=control_params)
	z_m =tf.reshape(z_m,[-1,dim])
	obs_params,params_e=computeEmission(z_holder,n_steps,dim,dim_emit,params=None,control_params=control_params)
	obs_params2,params_e=computeEmission(z_m,n_steps,dim,dim_emit,params=params_e,control_params=control_params)
	#init = tf.global_variables_initializer()
	#sess.run(init)
	print("[LOAD] ",config["load_model"])
	saver = tf.train.Saver()# これ以前の変数のみ保存される
	saver.restore(sess,config["load_model"])
	z0=np.zeros((batch_size*n_steps,dim),dtype=np.float32)
	#
	
	sim_nsteps=100
	data=np.zeros((batch_size*n_steps,sim_nsteps,dim),dtype=np.float32)
	obs_data=np.zeros((batch_size*n_steps,sim_nsteps,dim_emit),dtype=np.float32)
	z0=(np.random.random((batch_size*n_steps,dim))-0.5)*2
	#
	data[:,0,:]=z0[:,:]
	feed_dict={z_holder:z0}
	obs=sess.run(obs_params,feed_dict=feed_dict)
	o=obs[0].reshape((-1,dim_emit))
	obs_data[:,0,:]=o[:,:]
	#
	for i in range(sim_nsteps-1):
		feed_dict={z_holder:z0}
		m=sess.run(z_m,feed_dict=feed_dict)
		obs=sess.run(obs_params2,feed_dict=feed_dict)

		m=m.reshape((-1,dim))
		data[:,i+1,:]=m[:,:]
		o=obs[0].reshape((-1,dim_emit))
		obs_data[:,i+1,:]=o[:,:]
		d=z0-m
		e=np.sum(d**2,axis=1)
		z0=m
	n=m.shape[0]
	for i in range(n):
		for j in range(i):
			d=np.sum((m[i,:]-m[j,:])**2)
			if d>1.0e-10:
				print(i,j,d)
				pass
	## save results
	if config["simulation_path"]!="":
		sim_filename=config["simulation_path"]+"/infer.jbl"
		results={}
		results["z"]=data
		results["x"]=obs_data
		print("[SAVE] result : ",sim_filename)
		joblib.dump(results,sim_filename)



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
	#config=get_default_config()
	config={}
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
	#with tf.Graph().as_default():
	with tf.Graph().as_default(), tf.device('/cpu:0'):
		with tf.Session() as sess:
			# mode
			if args.model is not None:
				config["load_model"]=args.model
			if args.mode=="infer":
				infer(sess,config)
			elif args.mode=="field":
				field(sess,config)
			elif args.mode=="potential":
				potential(sess,config)
			else:
				infer(sess,config)

#python attractor.py --config hyopt20170704/config_infer.json --hyperparam hyopt20170704/hyparam00027.result.json
