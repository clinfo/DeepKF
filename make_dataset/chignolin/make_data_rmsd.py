#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import mdtraj as md
import numpy
from pylab import *
from math import pi
import sys
import glob

from msmbuilder.featurizer import ContactFeaturizer
from msmbuilder.featurizer import RMSDFeaturizer


#    argvs = sys.argv
#    fname1 = str(argvs[1])
#    oname = str(argvs[2])
#

list = ["0_31410"]
for name in list:
    topology = md.load(name + "/" + name + ".gro").topology
    print(topology)
    table, bonds = topology.to_dataframe()

    traj = name + "/protein_gpu/equil_n1/md1_noSOL_fit.xtc"
    t = md.load_xtc(traj, top=name + "/" + name + ".gro")

    ref_top = md.load("../0_0.gro")
    print(ref_top)

    # rmsd_featurizer
    rmsd_feat = RMSDFeaturizer(ref_top)
    rmsd = rmsd_feat.fit_transform(t)

    rmsd_arr = np.array(rmsd)
    print(rmsd_arr.shape)

    ####input形式：(100, 100, 1)#
    ####input形式：(500, 1000, 30)#
    data = np.delete(rmsd_arr, 0, 0)
    data2 = np.reshape(data, (100, -1, 1))
    print(data2.shape)

    np.save(name + "rmsd_p1_skip0_10ns.npy", data2)
