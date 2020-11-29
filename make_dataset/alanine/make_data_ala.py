#!/usr/bin/env python
# -*- coding: utf-8 -*-

from get_angles import get_angles
import numpy as np
import sys

if __name__ == "__main__":
    argvs = sys.argv
    filepath = '/data/traj_data/ala_data/'
    fname1_list = [filepath + 'trajectory-'+str(i)+'.dcd' for i in range(2, 9)]
    fname2_list = [filepath + 'trajectory-'+str(g)+'.xtc' for g in range(1, 6)]
    angles = get_angles(filepath + 'trajectory-1.dcd')
    for fname1 in fname1_list:
        angles_new = get_angles(fname1)
        angles = np.concatenate([angles, angles_new], axis=0)
    for fname2 in fname2_list:
        angles_new = get_angles(fname2)
        angles = np.concatenate([angles, angles_new], axis=0)
    print(angles.shape)   ##(180005, 2)
    angles = np.delete(angles, 0, 0)
    angles = np.delete(angles, 1, 0)
    angles = np.delete(angles, 2, 0)
    angles = np.delete(angles, 3, 0)
    angles = np.delete(angles, 4, 0)
    print(angles.shape)   ##(180000, 2)

    traj1 = np.reshape(angles, (-1, 300, 2))
    print(traj1.shape) #(600, 300, 2)
    np.save('ala_traj_all.npy', traj1)
