# -*- coding: utf-8 -*-

import sys
import numpy as np
import data_generator
import matplotlib.pyplot as plt

#%matplotlib inline

datapoints = int(1e6)
stride = 10

x = data_generator.get_folding_model_data(
    datapoints, rvec0=2.0 * (np.random.rand(2) - 0.5), kT=1.0, dt=0.1
)
r = np.linalg.norm(x, axis=-1)[::stride]

pot = np.zeros_like(r)
for i in range(r.shape[0]):
    pot[i] = data_generator.folding_model_energy(r[i], 3)


plt.plot(r[::stride], pot[::stride], ".")
plt.xlabel("Distance x / a.u.", fontsize=16)
plt.ylabel("Pot. energy / a.u.", fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.show()
plt.savefig("folding_energy.png")

# print(np.random.rand(2).shape) #(2, )
# print(np.zeros((2)).shape) #(2,)
#
#
# rvec = np.zeros(shape=(datapoints+1, 2))
# print(rvec.shape) #(1e6, 2)
# print(rvec[0, :].shape) #(2,)
#
#
traj_whole = x
traj_data_points, input_size = traj_whole.shape

print(traj_whole)
print(traj_whole.shape)

traj_whole = np.delete(traj_whole, 0, 0)
traj1 = np.reshape(traj_whole, (-1, 1000, 2))
print(traj1.shape)

np.save("folding_2d_traj.npy", traj1)
