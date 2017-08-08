import numpy as np
from matplotlib import pylab as plt
import joblib
import json
import sys
import os

filename_result="result/test.jbl"
filename_obs="pack_data_emit_test.npy"
filename_mask="pack_mask_emit_test.npy"
filename_info="pack_info.json"
pid_key="pid_list_test"
out_dir="plot_test"

#filename_result="result_dim66/train.jbl"
#filename_obs="pack_data_emit.npy"
#filename_mask="pack_mask_emit.npy"
#filename_info="pack_info.json"
#pid_key="pid_list_train"
#out_dir="plot_train"

if len(sys.argv)>1:
	fp = open(sys.argv[1], 'r')
	config=json.load(fp)

	filename_result=config["save_result_test"]
	filename_obs=config["data_test_npy"]
	filename_mask=config["mask_test_npy"]
	filename_steps=config["steps_test_npy"]
	filename_info=config["data_info_json"]
	if "plot_path" in config:
		out_dir=config["plot_path"]
		try:
			os.makedirs(out_dir)
		except:
			pass

print("[LOAD]:",filename_result)
obj=joblib.load(filename_result)
o=np.load(filename_obs)
print("[LOAD]:",filename_steps)
steps=np.load(filename_steps)
o=o.transpose((0,2,1))
print("[LOAD]:",filename_mask)
m=np.load(filename_mask)
m=m.transpose((0,2,1))
print("[LOAD]:",filename_info)
fp = open(filename_info, 'r')
data_info = json.load(fp)
d=data_info["attr_emit_list"].index("206010")
print("206010:",d)
o[m<0.1]=np.nan


z_q=obj["z_q"].reshape((-1,o.shape[1],2))
mu_q=obj["mu_q"]
cov_q=obj["cov_q"]
obs_mu=obj["obs_params"][0]
pred_mu=obj["pred_params"][0]
obs_mu[m<0.1]=np.nan
pred_mu[m<0.1]=np.nan
x=obj["x"]
print(mu_q.shape)
idx=702
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
	plt.plot(mu_q[idx,:s,0],label="dim0")
	plt.plot(mu_q[idx,:s,1],label="dim1")
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
