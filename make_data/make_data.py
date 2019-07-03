import sys
import numpy as np


data_arr =np.load("dkf_data/dkf_n1.npy")
print(data_arr.shape)
data_list =data_arr.tolist()
print(len(data_list))
print(len(data_list[0]))
print(len(data_list[0][0]))
data_size = len(data_list[0][0])

#for i in range(0, 500, 100):
#    data =data_arr[i:i+100,:,:].reshape((1, 100000, 30))
#    print(data.shape)
#    np.save("dkf_data/testdata/dkf_n1_"+str(int(i/100))+".npy",data)

#data = data_arr[260:300, :,:].reshape((1, 40000, 30))
#np.save("dkf_data/test_80ns_n1.npy",data)
#
#data = data_arr[320:360, :,:].reshape((1, 40000, 30))
#np.save("dkf_data/test_80ns_n2.npy",data)
#
#data = data_arr[420:460, :,:].reshape((1, 40000, 30))
#np.save("dkf_data/test_80ns_n3.npy",data)
#
#data = data_arr[480:500, :,:].reshape((1, 20000, 30))
#np.save("dkf_data/test_40ns_n1.npy",data)
#
#80ns_maskdata
mask_data=np.zeros((1, 20000, 30),np.int32)
for k in range(0, 20000, 500):
    if k%1000==0:
        for i in range(data_size):
            for j in range(k, k+500):
                mask_data[:,j,i]=1

print(mask_data)
np.save("dkf_data/mask_40ns_500step.npy", mask_data)
