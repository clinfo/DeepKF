import sys
import numpy as np


data_arr = np.load("dkf_data/dkf_n1.npy")
print(data_arr.shape)
data_list = data_arr.tolist()
print(len(data_list))
print(len(data_list[0]))
print(len(data_list[0][0]))
data_size = len(data_list[0][0])

data = data_arr[:150, :, :].reshape((1, 150000, 30))
print(data.shape)
# print(data)

mask_data = np.zeros((1, 150000, 30), np.int32)

print(mask_data.shape)

for k in range(0, 150000, 5000):
    #   print(k)
    if k % 10000 == 0:
        print(k)
        for i in range(data_size):
            for j in range(k, k + 5000):
                mask_data[:, j, i] = 1

print(mask_data)

np.save("dkf_data/mask_n2.npy", mask_data)
np.save("dkf_data/test_n2.npy", data)
