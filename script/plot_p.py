import numpy as np
import joblib
import json
import sys
import argparse
from plot_input import load_plot_data,get_default_argparser

parser=get_default_argparser()
parser.add_argument('--num_dim', type=int,
		default=2,
		help='the number of dim. (latent variables)')
parser.add_argument('--num_particle', type=int,
		default=10,
		help='the number of particles (latent variable)')
parser.add_argument('--obs_dim', type=int,
		default=0,
		help='a dimension of observation for plotting')
parser.add_argument('--obs_num_particle', type=int,
		default=10,
		help='the number of particles (observation)')
	
args=parser.parse_args()
# config

if not args.show:
	import matplotlib
	matplotlib.use('Agg')
	from matplotlib import pylab as plt
else:
	from matplotlib import pylab as plt
data=load_plot_data(args,result_key="save_result_filter")

#d=data.info["attr_emit_list"].index("206010")
#print("206010:",d)
idx=args.index
if data.mask is not None:
	data.obs[data.mask<0.1]=np.nan

#x=obj["x"]

z=data.result["z"]
#print("z:",z[0,idx,:,0])
print("z:",z.shape)
mu=data.result["mu"]

colorlist = ["g", "b", "r", "c", "m", "y", "k", "w"]

def plot_fig(idx):
	s=data.steps[idx]
	d=args.obs_dim
	print("data index:",idx)
	print("dimension (observation):",d)
	print("plotting dimension (latent): 0 -",args.num_dim-1)
	print("steps:",s)
	#print("z:",z[0,idx,:s,0])
	plt.subplot(3,1,1)
	for j in range(args.num_dim):
		plt.plot(z[0,idx,:s,j],label="dim"+str(j),color=colorlist[j])
		for i in range(args.num_particle-1):
			plt.plot(z[i+1,idx,:s,j],color=colorlist[j])
	plt.legend()
	
	plt.subplot(3,1,2)
	plt.plot(data.obs[idx,:s,d],label="x",color="b")
	plt.plot(mu[0,idx,:s,d],label="pred",color="g")
	for i in range(args.obs_num_particle):
		plt.plot(mu[i,idx,:s,d],color="g")
	mu2=np.mean(mu,axis=0)
	plt.plot(mu2[idx,:s,d],label="pred_mean",color="r")
	plt.legend()

	plt.subplot(3,1,3)
	errors=data.result["error"]
	plt.plot(errors[0,idx,:s,d],label="error",color="b")
	for i in range(args.obs_num_particle-1):
		plt.plot(errors[i+1,idx,:s,d],color="b")
	plt.legend()
	
if args.mode=="all":
	l=len(data.info[data.pid_key])
	if args.limit_all is not None and l > args.limit_all:
		l = args.limit_all
	for idx in range(l):
		name=data.info[data.pid_key][idx]
		#print(data.info[data.pid_key][idx])
		plot_fig(idx)
		out_filename=data.out_dir+"/"+str(idx)+"_"+name+"_p.png"
		print("[SAVE] :",out_filename)
		plt.savefig(out_filename)
		plt.clf()
else:
	idx=args.index
	#print(data.info[data.pid_key][idx])
	plot_fig(idx)
	out_filename=data.out_dir+"/"+str(idx)+"_p.png"
	print("[SAVE] :",out_filename)
	plt.savefig(out_filename)
	if args.show:
		plt.show()
	plt.clf()

