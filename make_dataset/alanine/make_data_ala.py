#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mdtraj as md
import numpy as np
import sys


def get_angles_dcd(fname1):
    traj = md.load_dcd(fname1, top="ala2.pdb")
    atoms, bonds = traj.topology.to_dataframe()
    psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]
    arr_angles = md.compute_dihedrals(traj, [phi_indices, psi_indices])
    return arr_angles


def get_angles_xtc(fname1):
    traj = md.load_xtc(fname1, top="ala2.pdb")
    atoms, bonds = traj.topology.to_dataframe()
    psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]
    arr_angles = md.compute_dihedrals(traj, [phi_indices, psi_indices])
    return arr_angles


if __name__ == "__main__":
    argvs = sys.argv
    fname1_list = ["../trajectory-" + str(i) + ".dcd" for i in range(2, 9)]
    fname2_list = ["../trajectory-" + str(g) + ".xtc" for g in range(1, 6)]
    angles = get_angles_dcd("../trajectory-1.dcd")
    for fname1 in fname1_list:
        angles_new = get_angles_dcd(fname1)
        angles = np.concatenate([angles, angles_new], axis=0)
    for fname2 in fname2_list:
        angles_new = get_angles_xtc(fname2)
        angles = np.concatenate([angles, angles_new], axis=0)
    print(angles.shape)  ##(80000, 2)

    angles = np.delete(angles, 0, 0)
    angles = np.delete(angles, 1, 0)
    angles = np.delete(angles, 2, 0)
    angles = np.delete(angles, 3, 0)
    angles = np.delete(angles, 4, 0)
    print(angles.shape)  ##(180000, 2)

    traj1 = np.reshape(angles, (-1, 300, 2))
    print(traj1.shape)  # (600, 300, 2)
    np.save("ala_traj_all.npy", traj1)
