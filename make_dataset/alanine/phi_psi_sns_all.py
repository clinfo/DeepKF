#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import mdtraj as md
import numpy
from pylab import *
from math import pi
import seaborn as sns
import pandas as pd


def get_angles_dcd(fname1):
    traj = md.load_dcd(fname1, top="../ala2.pdb")
    atoms, bonds = traj.topology.to_dataframe()
    psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]
    arr_angles = md.compute_dihedrals(traj, [phi_indices, psi_indices])
    return arr_angles


def get_angles_xtc(fname1):
    traj = md.load_xtc(fname1, top="../ala2.pdb")
    atoms, bonds = traj.topology.to_dataframe()
    psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]
    arr_angles = md.compute_dihedrals(traj, [phi_indices, psi_indices])
    return arr_angles


def get_ene(fname2):
    ene = []
    f = open(fname2, "r")
    text = f.readlines()
    for i in range(len(text)):
        itemList = text[i][:-1].split()
        if i >= 1:
            ene.append(float(itemList[3]))
    arr_ene = np.array(ene).reshape((10000, 1))
    return arr_ene


if __name__ == "__main__":
    argvs = sys.argv
    #    fname1 = str("../trajectory-1.dcd")
    #    fname2 = str(argvs[2])
    #    oname = str(argvs[3])
    for i in range(8):
        if i == 0:
            angles = get_angles_dcd("../trajectory-" + str(i + 1) + ".dcd")
            angle_df = pd.DataFrame(data=angles, columns=["Phi", "Psi"])
        if i >= 1:
            angles = get_angles_dcd("../trajectory-" + str(i + 1) + ".dcd")
            angle_df_new = pd.DataFrame(data=angles, columns=["Phi", "Psi"])
            angle_df = pd.concat([angle_df, angle_df_new], axis=0)
    print(angle_df)
    for i in range(5):
        angles = get_angles_xtc("../trajectory-" + str(i + 1) + ".xtc")
        angle_df_new = pd.DataFrame(data=angles, columns=["Phi", "Psi"])
        angle_df = pd.concat([angle_df, angle_df_new], axis=0)
    print(angle_df)
    #    ene = get_ene(fname2)

    sns_plot = sns.jointplot("Phi", "Psi", data=angle_df, kind="hex")
    sns_plot.savefig("sns_all_out.png")
