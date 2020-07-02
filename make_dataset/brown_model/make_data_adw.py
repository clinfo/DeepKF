import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline
import data_generator


## generate 50000 frames and energy values
datapoints = int(5e4)

#print(datapoints) ## 50000

traj_whole = data_generator.get_asymmetric_double_well_data(datapoints)
#for i in range(len(traj_whole)):
#    print(i,traj_whole[i])
# To fit the dataformat

plt.hist(traj_whole, bins=16)
plt.savefig("hist1.png")
"""
traj_whole = np.expand_dims(traj_whole, 1)
traj_data_points, input_size = traj_whole.shape

print(traj_whole.shape)  ## (50000, 1)

traj_whole = np.delete(traj_whole, 0, 0)
print(traj_whole)
traj1 = np.reshape(traj_whole, (-1, 100, 1))
print(traj1.shape)

traj2 = np.reshape(traj_whole, (100, -1, 1))
print(traj2.shape)

np.save("adw_traj1.npy", traj1)
# np.save('assym_traj2.npy', traj2)


x = np.linspace(-1, 5, 500)
plt.figure(figsize=(6, 4))
plt.ylim(-15, 10)
plt.xlim(-1, 5)
plt.plot(x, data_generator.asymmetric_double_well_energy(x), lw=2)
plt.xlabel("Position x / a.u.", fontsize=6)
plt.ylabel("Pot. energy / a.u.", fontsize=6)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
#plt.show()
plt.savefig("adw_potential.png")
"""