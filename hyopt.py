#
# pylint: disable=missing-docstring
from __future__ import print_function

import os
import re
import sys
import json

import tensorflow as tf
import numpy as np

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



hyperparameter=None

def initialize_hyperparameter(load_filename):
	global hyperparameter
	hyperparameter={}
	"""
	hyperparameter["evaluation"]=None
	hyperparameter["evaluation_output"]=None
	hyperparameter["hyperparameter_input"]=load_filename
	hyperparameter["emission_internal_layers"]=None
	hyperparameter["transition_internal_layers"]=None
	hyperparameter["variational_internal_layers"]=None
	hyperparameter["potential_internal_layers"]=None
	hyperparameter["potential_enabled"]=True
	hyperparameter["potential_grad_transition_enabled"]=True
	hyperparameter["potential_nn_enabled"]=True
	"""
	if load_filename is not None:
		fp = open(load_filename, 'r')
		hyperparameter.update(json.load(fp))
	
#
def get_hyperparameter():
	global hyperparameter
	if hyperparameter is None:
		initialize_hyperparameter(None)
	return hyperparameter


def save_hyperparameter(save_filename=None):
	global hyperparameter
	if hyperparameter is not None:
		if save_filename is None:
			save_filename=hyperparameter["evaluation_output"]
		#
		if save_filename is not None:
			print("[SAVE] hyperparameter: ",save_filename)
			fp = open(save_filename, "w")
			json.dump(hyperparameter, fp, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '),cls=NumPyArangeEncoder)



