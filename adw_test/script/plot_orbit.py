import numpy as np
from matplotlib import pylab as plt
import joblib
from matplotlib import animation
import json
import sys

filename_result="result/test.jbl"
filename_obs="data/pack_data_emit_test.npy"
filename_mask="data/pack_mask_emit_test.npy"
filename_info="data/pack_info.json"
pid_key="pid_list_test"
out_dir="plot_test"

if len(sys.argv)>1:
	fp = open(sys.argv[1], 'r')
	config=json.load(fp)
	filename_result=config["save_result_test"]
	filename_obs=config["data_test_npy"]
	filename_mask=config["mask_test_npy"]
	filename_steps=config["steps_test_npy"]
	if "plot_path" in config:
		out_dir=config["plot_path"]


steps=np.load(filename_steps)
pid_key="pid_list_test"
obj=joblib.load(filename_result)
z_q=obj["z_q"].reshape(obj["mu_q"].shape)
mu_q=obj["mu_q"]
cov_q=obj["cov_q"]
obs_mu=obj["obs_params"][0]
x2=obj["x"]
#
fp = open(filename_info, 'r')
data_info = json.load(fp)
d=data_info["attr_emit_list"].index("206010")
idx=3
l=len(data_info[pid_key])
def plot_mv(idx):
	s=int(steps[idx])
	print(s)
	x=mu_q[idx,:s,0]
	y=mu_q[idx,:s,1]
	fig=plt.figure()
	plt.subplot(2,1,1)
	plt.plot(x[1],y[1],"ro")
	plt.plot(x[1:s],y[1:s],label="mu_q")
	plt.legend()
	line, = plt.plot([], [],"ro", lw=2)

	plt.subplot(2,1,2)
	line2_x, = plt.plot([],[], lw=1,label="x")
	line2_o, = plt.plot([],[], lw=2,label="obs_mu")
	x_window=50
	plt.xlim(0,x_window)
	plt.ylim(-1.5,1.5)
	plt.legend()
	def init():
		line.set_data([], [])
		return line,
	def animate(i):
		if x_window+i<x2.shape[1]:
			px=x[i]
			py=y[i]
			line.set_data([px,px],[py,py])
			line2_x.set_data(range(x_window),x2[idx,i:x_window+i,d])
			line2_o.set_data(range(x_window),obs_mu[idx,i:x_window+i,d])
			return line,line2_x,line2_o
		return None

	anim = animation.FuncAnimation(fig, animate, init_func=init,
			frames=s, interval=500, blit=True)
	return anim

print(data_info[pid_key][idx])
name=data_info[pid_key][idx]
anim=plot_mv(idx)
anim.save(out_dir+"/"+str(idx)+"_"+name+'_orbit.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
#plt.show()
plt.clf()
