import numpy as np
import joblib
import json
import sys
import os
from matplotlib.colors import LinearSegmentedColormap

def generate_cmap(colors):
		values = range(len(colors))
		vmax = np.ceil(np.max(values))
		color_list = []
		for v, c in zip(values, colors):
				color_list.append( ( v/ vmax, c) )
		return LinearSegmentedColormap.from_list('custom_cmap', color_list)
###
def draw_heatmap(h1,h2,cmap,attr):
	plt.figure(figsize=(16, 6))
	plt.subplot(1, 1, 1)
	plt.imshow(h1, aspect='auto', interpolation='none',
					cmap=cmap, vmin=-1.0, vmax=1.0)#
	plt.gca().xaxis.set_ticks_position('none')
	plt.gca().yaxis.set_ticks_position('none')
	if h2 is not None:
		plt.subplot(2, 1, 2)
		plt.imshow(h2, aspect='auto', interpolation='none',
						cmap=cmap, vmin=-1.0, vmax=1.0)#
		plt.gca().xaxis.set_ticks_position('none')
		plt.gca().yaxis.set_ticks_position('none')


if len(sys.argv)>2 and sys.argv[2]=="all":
	import matplotlib
	matplotlib.use('Agg')
	from matplotlib import pylab as plt
else:
	from matplotlib import pylab as plt


filename_result="result/test.jbl"
filename_obs="pack_data_emit_test.npy"
filename_mask=None
filename_info=None
pid_key="pid_list_test"
out_dir="plot_test"

if len(sys.argv)>1:
	fp = open(sys.argv[1], 'r')
	config=json.load(fp)
	filename_result=config["save_result_test"]
	filename_obs=config["data_test_npy"]
	if "mask_test_npy" in config:
		filename_mask=config["mask_test_npy"]
	filename_steps=config["steps_test_npy"]
	filename_info=config["data_info_json"]
	if "plot_path" in config:
		out_dir=config["plot_path"]
		os.makedirs(out_dir,exist_ok=True)

print("[LOAD]:",filename_result)
obj=joblib.load(filename_result)
print("[LOAD]:",filename_obs)
o=np.load(filename_obs)
o=o.transpose((0,2,1))
print("[LOAD]:",filename_steps)
steps=np.load(filename_steps)
if filename_mask is None:
	print("[LOAD]:",filename_mask)
	m=np.load(filename_mask)
	m=m.transpose((0,2,1))

print("[LOAD]:",filename_info)
fp = open(filename_info, 'r')
data_info = json.load(fp)
#d=data_info["attr_emit_list"].index("206010")
#print("206010:",d)
d=0
o[m<0.1]=np.nan

print(obj.keys())

# z
z_q=obj["z_q"]
print(z_q.shape)
mu_q=obj["mu_q"]
# obs
obs_mu=obj["obs_params"][0]
pred_mu=obj["pred_params"][0]
print(m.shape)
print(obs_mu.shape)
obs_mu[m<0.1]=np.nan
pred_mu[m<0.1]=np.nan

print(mu_q.shape)
idx=0
l=len(data_info[pid_key])

def plot_fig(idx):
	print("id   =",idx)
	print("data =",data_info[pid_key][idx])
	print("error=",np.nanmean((obs_mu[:,:-1,:]-o[:,1:,:])**2))
	s=steps[idx]
	plt.subplot(2,1,2)
	plt.plot(o[idx,:s,d],label="x")
	plt.plot(obs_mu[idx,:s,d],label="recons")
	plt.plot(pred_mu[idx,:s,d],label="pred")
	plt.legend()
	plt.subplot(2,1,1)
	if False:
		plt.plot(mu_q[idx,:s,0],label="dim0")
		plt.plot(mu_q[idx,:s,1],label="dim1")
	else:
		cmap = generate_cmap(['#0000FF','#FFFFFF','#FF0000'])
		h=z_q[idx,:s,:]
		#h=mu_q[idx,:s,:]
		print(s)
		print(h.shape)
		plt.imshow(np.transpose(h), aspect='auto', interpolation='none',
					cmap=cmap, vmin=-1.0, vmax=1.0)#
		plt.gca().xaxis.set_ticks_position('none')
		plt.gca().yaxis.set_ticks_position('none')
	plt.legend()

name=data_info[pid_key][idx]
if len(sys.argv)>2:
	if sys.argv[2]=="all":
		for idx in range(l):
			plot_fig(idx)
			out_filename=out_dir+"/"+str(idx)+"_"+name+"_plot.png"
			print("[SAVE] :",out_filename)
			plt.savefig(out_filename)
			plt.clf()
	else:
		idx=int(sys.argv[2])
		plot_fig(idx)
		out_filename=out_dir+"/"+str(idx)+"_"+name+"_plot.png"
		print("[SAVE] :",out_filename)
		plt.savefig(out_filename)
		plt.show()
		plt.clf()
else:
	plot_fig(idx)
	out_filename=out_dir+"/"+str(idx)+"_"+name+"_plot.png"
	print("[SAVE] :",out_filename)
	plt.savefig(out_filename)
	plt.show()
	plt.clf()
