import numpy as np
import joblib
import json
import sys
import argparse
from plot_input import load_plot_data,get_default_argparser
parser=get_default_argparser()
args=parser.parse_args()
# config

if args.mode=="all":
	import matplotlib
	matplotlib.use('Agg')
	from matplotlib import pylab as plt
else:
	from matplotlib import pylab as plt
data=load_plot_data(args,result_key="save_result_filter")

#d=data.info["attr_emit_list"].index("206010")
#print("206010:",d)
d=0
idx=0
if data.mask:
	data.obs[data.mask<0.1]=np.nan

#x=obj["x"]
l=len(data.info[data.pid_key])

z=data.result["z"]
print("z:",z[0,idx,:,0])
print(z.shape)
mu=data.result["mu"]

def plot_fig(idx):
	s=data.steps[idx]
	print("steps:",s)
	print("z:",z[0,idx,:s,0])
	
	plt.subplot(3,1,1)
	plt.plot(z[0,idx,:s,0],label="dim0",color="b")
	for i in range(10-1):
		plt.plot(z[i+1,idx,:s,0],color="b")
	plt.plot(z[0,idx,:s,1],label="dim1",color="g")
	for i in range(10-1):
		plt.plot(z[i+1,idx,:s,1],color="g")
	plt.legend()
	
	plt.subplot(3,1,2)
	plt.plot(data.obs[idx,:s,d],label="x",color="b")
	plt.plot(mu[0,idx,:s,d],label="pred",color="g")
	for i in range(10):
		plt.plot(mu[i,idx,:s,d],color="g")
	mu2=np.mean(mu,axis=0)
	plt.plot(mu2[idx,:s,d],label="pred_mean",color="r")
	plt.legend()

	plt.subplot(3,1,3)
	errors=data.result["error"]
	plt.plot(errors[0,idx,:s,d],label="x",color="b")
	
if args.mode=="all":
	for idx in range(l):
		name=data.info[data.pid_key][idx]
		print(data.info[data.pid_key][idx])
		name=data.info[data.pid_key][idx]
		plot_fig(idx)
		out_filename=data.out_dir+"/"+str(idx)+"_"+name+"_p.png"
		print("[SAVE] :",out_filename)
		plt.savefig(out_filename)
		plt.clf()
else:
	idx=args.index
	print(data.info[data.pid_key][idx])
	plot_fig(idx)
	out_filename=data.out_dir+"/"+str(idx)+"_"+name+"_p.png"
	print("[SAVE] :",out_filename)
	plt.savefig(out_filename)
	plt.show()
	plt.clf()

