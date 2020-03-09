import sys
import numpy as np

data=[]
mask=[]
print("[LOAD] data/sample.csv")
fp=open("sample/sample.csv")
line = fp.readline()
num=int(line.strip())

for line in fp.readlines():
	s=line.strip()
	arr=s.split(",")
	vec=[]
	mask_vec=[]
	for el in arr:
		e=el.strip()
		m=0
		if e!="":
			m=1
			f=float(e)
		else:
			f=0
			#f=np.nan
		vec.append(f)
		mask_vec.append(m)
	data.append(vec)
	mask.append(mask_vec)
#print(data)
#print(mask)
max_length=max(map(len,data))
num_data=int(len(data)/num)
#print(max_length)
#print(num_data,num,max_length)
mat_data=np.zeros((num_data,num,max_length),np.float32)
mat_mask=np.zeros((num_data,num,max_length),np.int32)
sep_data=zip(*[iter(data)]*num)
sep_mask=zip(*[iter(mask)]*num)
steps=[]
for i,pair in enumerate(zip(sep_data,sep_mask)):
	d=np.array(pair[0])
	m=np.array(pair[1])
	mat_data[i,:d.shape[0],:d.shape[1]]=d
	mat_mask[i,:m.shape[0],:m.shape[1]]=m
	steps.append(d.shape[1])

#print(mat_data)
#print(mat_mask)
mat_steps = np.array(steps,np.int32)

print("[SAVE] data/sample_data.npy")
print("[SAVE] data/sample_mask.npy")
print("[SAVE] data/sample_steps.npy")
np.save("data/sample_data.npy",mat_data)
np.save("data/sample_mask.npy",mat_mask)
np.save("data/sample_steps.npy",mat_steps)

