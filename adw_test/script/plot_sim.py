import numpy as np
from matplotlib import pylab as plt
import joblib
from matplotlib import animation
import json
import sys





filename_info="data/pack_info.json"
filename_result="sim.jbl"
pid_key="pid_list_test"
out_dir="plot_test"

if len(sys.argv)>1:
	fp = open(sys.argv[1], 'r')
	config=json.load(fp)
	if "plot_path" in config:
		out_dir=config["plot_path"]
		filename_result=config["simulation_path"]+"/infer.jbl"


print("[LOAD] ",filename_result)
obj=joblib.load(filename_result)
data_x=obj["x"]
data_z=obj["z"]
print(data_x.shape)
print(data_z.shape)
#
fp = open(filename_info, 'r')
data_info = json.load(fp)
d=data_info["attr_emit_list"].index("206010")
idx=3
s=data_x.shape[1]
l=len(data_info[pid_key])
num=100
def plot_mv(idx):
	x=data_z[:num,:,0]
	y=data_z[:num,:,1]
	fig=plt.figure()
	plt.subplot(2,1,1)
	idx=0
	plt.plot(x[idx,:],y[idx,:],color="b",label="z")
	for idx in range(num):
		plt.plot(x[idx,0],y[idx,0],"ro")
		plt.plot(x[idx,:],y[idx,:],color="b")
	plt.legend()
	points=[]
	for idx in range(num):
		line, = plt.plot([], [],"ro", lw=2)
		points.append(line)
	
	plt.subplot(2,1,2)
	line2_x, = plt.plot([],[], lw=1,label="x")
	x_window=50
	plt.xlim(0,x_window)
	plt.ylim(-1.5,1.5)
	plt.legend()
	def init():
		for line in points:
			line.set_data([], [])
		return points
	def animate(i):
		if i<data_x.shape[1]:
			for idx,line in enumerate(points):
				px=x[idx,i]
				py=y[idx,i]
				line.set_data([px,px],[py,py])
		idx=0
		xx=np.zeros((x_window),dtype=np.float)
		xx[:]=np.nan
		n=data_x[idx,i:x_window+i,d].shape[0]
		xx[:n]=data_x[idx,i:x_window+i,d]
		line2_x.set_data(range(x_window),xx)
		return line,line2_x
	#return None

	anim = animation.FuncAnimation(fig, animate, init_func=init,
			frames=100, interval=500, blit=True)
	return anim

name="sim"
anim=plot_mv(idx)
out_filename=out_dir+"/"+str(idx)+"_"+name+".mp4"
print("[SAVE] :",out_filename)
#anim.save('test.mp4', fps=10, extra_args=['-vcodec', 'libx264'])
anim.save(out_filename, fps=10, extra_args=['-vcodec', 'libx264'])
plt.show()
plt.clf()
