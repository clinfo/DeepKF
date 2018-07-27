import numpy as np
import joblib
import json
import sys
import os
import argparse

class dotdict(dict):
	"""dot.notation access to dictionary attributes"""
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

def get_default_argparser():
	parser = argparse.ArgumentParser()
	parser.add_argument('mode', type=str,
			help='test/all')
	parser.add_argument('--config', type=str,
			default=None,
			nargs='?',
			help='config json file')
	parser.add_argument('--hyperparam', type=str,
			default=None,
			nargs='?',
			help='config json file')
	parser.add_argument('--info_key', type=str,
			default="pid_list_test",
			nargs='?',
			help='the key of the info file for data list, e.g., pid_list_test/pid_list_train')
	parser.add_argument('--out_dir', type=str,
			default="plot_test",
			help='output directory for images')
	parser.add_argument('--index', type=int,
			default=0,
			help='data index of plotting target (only test mode)')
	parser.add_argument('--show',
			action='store_true',
			default=False,
			help='data index of plotting target (only test mode)')
	return parser

def make_default_info(x):
	info={}
	info["pid_list_test"]=["data"+str(i)  for i in range(x.shape[0])]
	info["attr_emit_list"]=["attr"+str(i)  for i in range(x.shape[2])]
	return info
def get_param(config,key,default_value):
	if config is not None and key in config:
		return config[key]
	return default_value
	
def load_plot_data(args,result_key="save_result_test"):
	print("====",result_key)
	if args.config is None:
		parser.print_help()
		quit()
	fp = open(args.config, 'r')
	config=json.load(fp)
	if args.hyperparam=="":
		fp = open(args.hyperparam, 'r')
		config.update(json.load(fp))
	pid_key=args.info_key
	out_dir=args.out_dir
	filename_result=get_param(config,result_key,None)
	filename_obs   =get_param(config,"data_test_npy","pack_data_emit_test.npy")
	filename_mask  =get_param(config,"mask_test_npy",None)
	filename_info  =get_param(config,"data_info_json",None)
	filename_steps =get_param(config,"steps_test_npy",None)
	out_dir        =get_param(config,"plot_path", out_dir)
	if out_dir:
		os.makedirs(out_dir,exist_ok=True)
	data=dotdict({})
	data.out_dir=out_dir
	data.pid_key=pid_key

	print("[LOAD]:",filename_result)
	data.result=joblib.load(filename_result)
	print("[LOAD]:",filename_obs)
	o=np.load(filename_obs)
	if "time_major" in config and not config["time_major"]:
		o=o.transpose((0,2,1))
	data.obs=o
	
	data.steps=None
	if filename_steps:
		print("[LOAD]:",filename_steps)
		data.steps=np.load(filename_steps)
	else:
		s=[len(data.obs[i]) for i in range(len(data.obs))]
		data.steps=np.array(s)
	data.mask=None
	if filename_mask is not None:
		print("[LOAD]:",filename_mask)
		m=np.load(filename_mask)
		if not config["time_major"]:
			m=m.transpose((0,2,1))
		data.mask=m

	if filename_info:
		print("[LOAD]:",filename_info)
		fp = open(filename_info, 'r')
		data.info = json.load(fp)
	else:
		data.info=make_default_info(data.obs)
	return data

