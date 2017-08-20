#
# pylint: disable=missing-docstring
from __future__ import print_function

import os
import re
import sys
import json

import tensorflow as tf
import numpy as np

hyperparameter=None

def initialize_hyperparameter(load_filename):
	global hyperparameter
	hyperparameter={}
	hyperparameter["evaluation"]=None
	hyperparameter["evaluation_output"]=None
	hyperparameter["hyperparameter_input"]=load_filename
	hyperparameter["emssion_internal_layers"]=[
			{"name":"fc"}
			]
	hyperparameter["transition_internal_layers"]=[
			{"name":"fc"},
			{"name":"fc"}
			]
	hyperparameter["variational_internal_layers"]=[
			{"name":"fc"},
			{"name":"lstm"}
			]
	hyperparameter["potential_internal_layers"]=[
			{"name":"fc"},
			{"name":"do"},
			{"name":"fc"},
			{"name":"do"},
			{"name":"fc"}
			]

	hyperparameter["potential_enabled"]=True
	hyperparameter["potential_grad_transition_enabled"]=True
	hyperparameter["potential_nn_enabled"]=True
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
			json.dump(hyperparameter, fp, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))



