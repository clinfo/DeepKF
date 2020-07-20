import numpy as np
import matplotlib.pyplot as plt
import data_generator

datapoints = int(100000)

traj_whole = data_generator.get_asymmetric_double_well_data(datapoints)
mask = [1.0]*10000 + [0.0]*40000

data_train_npy = np.expand_dims(traj_whole[0:50000], 1)
data_test_npy = np.expand_dims(traj_whole[50000:100000], 1)
mask_test_npy = np.expand_dims(mask, 1)

data_train_npy = np.reshape(data_train_npy, (-1, 100, 1))
data_test_npy = np.reshape(data_test_npy, (-1, 100, 1))
mask_test_npy = np.reshape(mask_test_npy, (-1, 100, 1))

np.save('data_train.npy', data_train_npy)
print(data_train_npy.shape)
np.save('data_test.npy', data_test_npy)
print(data_test_npy.shape)
np.save('mask_test.npy', mask_test_npy)
print(mask_test_npy.shape)
print(mask_test_npy)
