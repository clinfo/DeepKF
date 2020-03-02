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



#    argvs = sys.argv
#    fname1 = str(argvs[1])
#    oname = str(argvs[2])
#
list=["0_31410", "0_31414", "0_31418"]
for name in list:
    topology = md.load(name+'/'+name+'.gro').topology
    print(topology)

    table, bonds = topology.to_dataframe()
    print(table.head())

    print(topology.atom(46))# ASP3-CA
    print(topology.atom(44))# ASP3-N
    print(topology.atom(119))# THR8-0
    print(topology.atom(105))# GLY7-O


    atom_pairs=np.array([[44, 105], [44, 119]])

    traj=name+'/protein_gpu/equil_n1/md1_noSOL_fit_skip.xtc'
    t = md.load_xtc(traj, top=name+'/'+name+'.gro')

    dist= md.compute_distances(t, atom_pairs, periodic=True, opt=True)

    data1 = np.array(dist)
    print(data1.shape)
#print(data1)
#np.save(str(i)+"_skip4_dist.npy", data1)
#

    plt.figure()
#    plt.title('Dihedral Map: Alanine dipeptide')
    plt.scatter(dist[:, 0], dist[:, 1], s=10, alpha=0.7, c=t.time)
    cbar = plt.colorbar()
    cbar.set_label('Time')
    plt.xlabel(r'Asp3N-Gly7O [nm]')
    plt.xlim(0, 2)
    plt.ylabel(r'Asp3N-Thr8O [nm]')
    plt.ylim(0, 2)
    plt.savefig(name+"_2dist_plot.png")
