import sys
import numpy as np
import data_generator

nstep = int(999999)

x_train = data_generator.get_folding_model_data(nstep=nstep,\
     rvec0=2.0 * (np.random.rand(2) - 0.5), kT=1., dt = 0.1)
print(x_train)
print(x_train.shape)
traj_train = np.reshape(x_train, (-1, 1000, 2))
print(traj_train.shape)
np.save("data_train.npy", traj_train)

x_test = data_generator.get_folding_model_data(nstep=nstep,\
     rvec0=2.0 * (np.random.rand(2) - 0.5), kT=1., dt = 0.1)
print(x_test)
print(x_test.shape)
traj_test = np.reshape(x_test, (-1, 1000, 2))
print(traj_test.shape)
np.save("data_test.npy", traj_test)
