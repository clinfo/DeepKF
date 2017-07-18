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
from dkf_model import computeTransition
from dkf_model import computeEmission
import hyopt as hy
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_boolean('use_fp16', False,"""Train the model using fp16.""")


def infer(sess,config):
	batch_size=config["batch_size"]
	if config["dim"] is None:
		dim=dim_emit
		config["dim"]=dim
	else:
		dim=config["dim"]
	batch_size=1000
	dim_emit=66
	n_steps=1
	z_holder=tf.placeholder(tf.float32,shape=(None,dim))
	
	# inference
	z_m,z_cov,_=computeTransition(z_holder,n_steps,dim,mean_prior0=None,cov_prior0=None,params=None)
	z_m =tf.reshape(z_m,[-1,dim])
	obs_params,params_e=computeEmission(z_holder,n_steps,dim,dim_emit,params=None)
	obs_params2,params_e=computeEmission(z_m,n_steps,dim,dim_emit,params=params_e)
	#init = tf.global_variables_initializer()
	#sess.run(init)
	saver = tf.train.Saver()# これ以前の変数のみ保存される
	saver.restore(sess,config["load_model"])
	z0=np.zeros((batch_size*n_steps,dim),dtype=np.float32)
	#
	
	sim_nsteps=100
	data=np.zeros((batch_size*n_steps,sim_nsteps,dim),dtype=np.float32)
	obs_data=np.zeros((batch_size*n_steps,sim_nsteps,dim),dtype=np.float32)
	z0=(np.random.random((batch_size*n_steps,dim))-0.5)*2
	#
	data[:,0,:]=z0[:,:]
	feed_dict={z_holder:z0}
	obs=sess.run(obs_params,feed_dict=feed_dict)
	o=obs[0].reshape((-1,dim))
	obs_data[:,0,:]=o[:,:]
	#
	for i in range(sim_nsteps-1):
		feed_dict={z_holder:z0}
		m=sess.run(z_m,feed_dict=feed_dict)
		obs=sess.run(obs_params2,feed_dict=feed_dict)

		m=m.reshape((-1,dim))
		data[:,i+1,:]=m[:,:]
		o=obs[0].reshape((-1,dim))
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
	## save results
	config["save_simulation"]="sim.jbl"
	if config["save_simulation"]!="":
		results={}
		results["z"]=data
		results["x"]=obs_data
		print("[SAVE] result : ",config["save_simulation"])
		joblib.dump(results,config["save_simulation"])



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
			else:
				infer(sess,config)

#python attractor.py --config hyopt20170704/config_infer.json --hyperparam hyopt20170704/hyparam00027.result.json
