# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from six.moves import xrange	# pylint: disable=redefined-builtin
import tensorflow as tf

def inputs(config,with_shuffle=True,with_train_test=True,test_flag=False):
	#x = np.load("test.npy")
	#m = np.load("mask.npy")

	#x = np.load("data_emit.npy")
	#m = np.load("mask_emit.npy")
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
	return tr_x,tr_m,te_x,te_m

