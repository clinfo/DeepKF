import numpy as np
import joblib
import json
import sys

if len(sys.argv)>2 and sys.argv[2]=="all":
	import matplotlib
	matplotlib.use('Agg')
	from matplotlib import pylab as plt
else:
	from matplotlib import pylab as plt




filename_info="data/pack_info.json"
filename_result="sim.jbl"
pid_key="pid_list_test"
out_dir="plot_test"

if len(sys.argv)>1:
	fp = open(sys.argv[1], 'r')
	config=json.load(fp)
	if "plot_path" in config:
		out_dir=config["plot_path"]
		filename_result=config["simulation_path"]+"/field.jbl"


print("[LOAD] ",filename_result)
obj=joblib.load(filename_result)
data_z=obj["z"]
data_gz=-obj["gz"][0]
print("shape z:",data_z.shape)
print("shape grad. z",data_gz.shape)
#
#fp = open(filename_info, 'r')
#data_info = json.load(fp)
#d=data_info["attr_emit_list"].index("206010")

X=data_z[:,0]
Y=data_z[:,1]
U=data_gz[:,0]
V=data_gz[:,1]
R=np.sqrt(U**2+V**2)
plt.axes([0.025, 0.025, 0.95, 0.95])
plt.quiver(X, Y, U, V, R, alpha=.5)
plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)
r=3.0#20
plt.xlim(-r, r)
#plt.xticks(())
plt.ylim(-r,r)
#plt.yticks(())

out_filename=out_dir+"/vec.png"
print("[SAVE] :",out_filename)
plt.savefig(out_filename)

plt.show()
plt.clf()
