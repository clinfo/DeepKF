import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

info={
	"attr_emit_list": ["item1"],
	"pid_list_train": [],
	"pid_list_test": []
	}
count=0
data_train=[]
for i in range(100):
	x2=np.random.normal(0,1,100)
	a=np.random.randint(50)
	x2[a:a+50]+=10
	x2=x2.reshape(-1,1)
	data_train.append(x2)
	info["pid_list_train"].append("data"+str(count))
	count+=1
filename="data_train.npy"
np.save(filename,np.array(data_train))

data_test=[]
for i in range(20):
	x2=np.random.normal(0,1,100)
	a=np.random.randint(50)
	x2[a:a+50]+=10
	x2=x2.reshape(-1,1)
	data_test.append(x2)
	info["pid_list_test"].append("data"+str(count))
	count+=1

filename="data_train.npy"
np.save(filename,np.array(data_train))
print("[SAVE]",filename)

filename="data_test.npy"
np.save(filename,np.array(data_test))
print("[SAVE]",filename)

filename="data_all.npy"
np.save(filename,np.array(data_train+data_test))
print("[SAVE]",filename)

fp=open("info.json","w")
json.dump(info,fp)
print("[SAVE]","info.json")
#obj={"0":"pos","1":"neg"}
#fp=open("synth.label.json","w")
#json.dump(obj,fp)

