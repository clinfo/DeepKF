import sys
import numpy as np


data_arr =np.load("md2_test.npy")
print(data_arr.shape)
data_list =data_arr.tolist()
print(len(data_list))
print(len(data_list[0]))
print(len(data_list[0][0]))
data_size = len(data_list[0][0])

#data =data_arr[:20,:5000 ,:].reshape((20, -1, 1))
#print(data.shape)
#np.save("rmsd_p1_skip0_10ns_200ns.npy",data)
##print(data)


#make maskdata
mask_data=np.zeros((500, 1000, 30),np.int32)
print(mask_data.shape)

for k in range(0, 1000, 100):
#   print(k)
    if k%200==0:
        print(k)
        for i in range(data_size):
            for j in range(k, k+100):
                mask_data[:, j, i]=1
    
print(mask_data)
np.save("md2_mask.npy",mask_data)

