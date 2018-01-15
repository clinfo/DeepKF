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


def load_data(config,with_shuffle=True,with_train_test=True,test_flag=False,output_dict_flag=True):
	if not test_flag:
		x = np.load(config["data_train_npy"])
		m = np.load(config["mask_train_npy"])
	else:
		x = np.load(config["data_test_npy"])
		m = np.load(config["mask_test_npy"])
	x=x.transpose((0,2,1))
	m=m.transpose((0,2,1))
	m2=np.zeros(m.shape[0:2],dtype=int)
	mm=np.sum(m,axis=2)
	m2[mm>0]=1
	# train / validatation/ test
	data_num=x.shape[0]
	data_idx=list(range(data_num))
	if with_shuffle:
		np.random.shuffle(data_idx)
	# split train/test
	sep=[0.0,1.0]
	if with_train_test:
		sep=config["train_test_ratio"]
	prev_idx=0
	sum_r=0.0
	sep_idx=[]
	for r in sep:
		sum_r+=r
		idx=int(data_num*sum_r)
		sep_idx.append([prev_idx,idx])
		prev_idx=idx
	print("#training data:",sep_idx[0][1]-sep_idx[0][0])
	print("#valid data:",sep_idx[1][1]-sep_idx[1][0])
	

	tr_x=x[sep_idx[0][0]:sep_idx[0][1]]
	tr_m=m2[sep_idx[0][0]:sep_idx[0][1]]
	te_x=x[sep_idx[1][0]:sep_idx[1][1]]
	te_m=m2[sep_idx[1][0]:sep_idx[1][1]]
	tr_x=tr_x[0:100]
	tr_m=tr_m[0:100]
	te_x=te_x[0:100]
	te_m=te_m[0:100]
	#return tr_x,tr_m,te_x,te_m
	train_data=dotdict({})
	train_data.x=tr_x
	train_data.m=tr_m
	train_data.num=tr_x.shape[0]
	train_data.n_steps=tr_x.shape[1]
	train_data.dim=tr_x.shape[2]
	valid_data=dotdict({})
	valid_data.x=te_x
	valid_data.m=te_m
	valid_data.num=te_x.shape[0]
	valid_data.n_steps=te_x.shape[1]
	valid_data.dim=tr_x.shape[2]
	return train_data,valid_data

