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

list=["0_31410"]
for name in list:
    topology = md.load(name+'/'+name+'.gro').topology
    print(topology)

    table, bonds = topology.to_dataframe()
#    print(table.head())


#プロットするための２距離をとる
#    print(topology.atom(46))# ASP3-CA
#    print(topology.atom(44))# ASP3-N
#    print(topology.atom(119))# THR8-0
#    print(topology.atom(105))# GLY7-O
#
#
#    atom_pairs=np.array([[44, 105], [44, 119]])

    traj=name+'/protein_gpu/equil_n1/md1_noSOL_fit.xtc'
    t = md.load_xtc(traj, top=name+'/'+name+'.gro')
    
    ref_top = md.load('../0_0.gro')
    print(ref_top)
#    dist= md.compute_distances(t, atom_pairs, periodic=True, opt=True)
#
#    dist_arr = np.array(dist)
#    print(dist_arr.shape)
#
#
##cantact_featurizer
#    contact_feat=ContactFeaturizer(contacts='all', scheme='closest-heavy', ignore_nonprotein=True)
#    contact=contact_feat.transform(t)
#
#    contact_arr = np.array(contact)
#    print(contact_arr.shape)
##        np.save(str(i)+"_skip2_data.npy", data1)
#

#rmsd_featurizer
    rmsd_feat = RMSDFeaturizer(ref_top)
    rmsd = rmsd_feat.fit_transform(t)
    
    rmsd_arr=np.array(rmsd)
    print(rmsd_arr.shape)

####input形式：(100, 100, 1)#
####input形式：(500, 1000, 30)#
    data=np.delete(rmsd_arr, 0, 0)
    data2=np.reshape(data, (100, -1, 1))
    print(data2.shape)
    
    np.save(name+"rmsd_p1_skip0_10ns.npy", data2)
