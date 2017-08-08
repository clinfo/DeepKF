import numpy as np
from matplotlib import pylab as plt
import joblib
import json
import sys

filename_result="result/test.jbl"
filename_obs="data/pack_data_emit_test.npy"
filename_mask="data/pack_mask_emit_test.npy"
filename_info="data/pack_info.json"
pid_key="pid_list_test"
out_dir="plot_test"
filename_filter="test.jbl"

if len(sys.argv)>1:
	fp = open(sys.argv[1], 'r')
	config=json.load(fp)

	filename_result=config["save_result_test"]
	filename_obs=config["data_test_npy"]
	filename_mask=config["mask_test_npy"]
	filename_filter=config["save_result_filter"]
	filename_steps=config["steps_test_npy"]
	out_dir=config["plot_path"]

print("[LOAD] ",filename_filter)
p_filter_result=joblib.load(filename_filter)
print("[LOAD] ",filename_result)
obj=joblib.load(filename_result)
print("[LOAD] ",filename_obs)
o=np.load(filename_obs)
o=o.transpose((0,2,1))
print("[LOAD] ",filename_steps)
steps=np.load(filename_steps)
print("[LOAD] ",filename_mask)
m=np.load(filename_mask)
m=m.transpose((0,2,1))
print("[LOAD] ",filename_info)
fp = open(filename_info, 'r')
data_info = json.load(fp)
d=data_info["attr_emit_list"].index("206010")
print("206010:",d)
o[m<0.1]=np.nan


x=obj["x"]
idx=302
l=len(data_info[pid_key])

z=p_filter_result["z"]
mu=p_filter_result["mu"]
mu[:,m<0.1]=np.nan
def plot_fig(idx):
	s=steps[idx]
	plt.subplot(2,1,1)
	plt.plot(z[0,idx,:s,0],label="dim0",color="b")
	for i in range(10-1):
		plt.plot(z[i+1,idx,:s,0],color="b")
	plt.plot(z[0,idx,:s,1],label="dim0",color="g")
	for i in range(10-1):
		plt.plot(z[i+1,idx,:s,1],color="g")
	#plt.legend()
	plt.subplot(2,1,2)
	plt.plot(o[idx,:s,d],label="x",color="b")
	plt.plot(mu[0,idx,:s,d],label="pred",color="g")
	for i in range(10):
		plt.plot(mu[i,idx,:s,d],color="g")
	mu2=np.mean(mu,axis=0)
	plt.plot(mu2[idx,:s,d],label="pred_mu",color="r")
	#plt.legend()

name=data_info[pid_key][idx]
if len(sys.argv)>2:
	if sys.argv[2]=="all":
		for idx in range(l):
			print(data_info[pid_key][idx])
			name=data_info[pid_key][idx]
			plot_fig(idx)
			out_filename=out_dir+"/"+str(idx)+"_"+name+"_p.png"
			print("[SAVE] :",out_filename)
			plt.savefig(out_filename)
			plt.clf()
	else:
		idx=int(sys.argv[2])
		print(data_info[pid_key][idx])
		plot_fig(idx)
		out_filename=out_dir+"/"+str(idx)+"_"+name+"_p.png"
		print("[SAVE] :",out_filename)
		plt.savefig(out_filename)
		plt.show()
		plt.clf()
else:
	print(data_info[pid_key][idx])
	plot_fig(idx)
	out_filename=out_dir+"/"+str(idx)+"_"+name+"_p.png"
	print("[SAVE] :",out_filename)
	plt.savefig(out_filename)
	plt.show()
	plt.clf()
