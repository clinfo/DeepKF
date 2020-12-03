#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
#from pylab import *
from math import pi
import sys
import glob

from msmbuilder.featurizer import ContactFeaturizer

def get_xy(fname):
    x=[]
    y=[]
    f= open(fname, "r")
    text= f.readlines()
    for i in range(len(text)):
        if i>=18:
            line =text[i][:-1].split()
            if len(line)==2:
                x.append(float(line[0]))
                y.append(float(line[1]))
    return x, y

dir ="/data/traj_data/chig_data/"
region="traj_n1"
topo_num="0_31410"
filepath=dir+region+'/'+topo_num+'/'

#初期構造トポロジカルデータをロード
topology = md.load(filepath+topo_num+'.gro').topology
print(topology)
table, _ = topology.to_dataframe()
print(table)
#print(bonds)

#トラジェクトリをロード
traj_xtc=filepath+'protein_gpu/equil_n1/md1_noSOL_fit.xtc'
traj = md.load_xtc(traj_xtc, top=filepath+topo_num+'.gro')

#トラジェクトリからASP3-NとTHR8-O, ASP3-NとGLY7-Oの距離を算出
print(topology.atom(44))# ASP3-N
print(topology.atom(119))# THR8-0
print(topology.atom(105))# GLY7-O
atom_pairs=np.array([[44, 105], [44, 119]])
distance = md.compute_distances(traj, atom_pairs, periodic=True, opt=True)
distance = np.array(distance)
print(distance)

#トラジェクトリ上の各点におけるポテンシャルエネルギーの値を取得
#energy_file=filepath+'/protein_gpu/equil_n1/energy.xvg'
#_, energy = get_xy(energy_file)

plt.figure()
plt.scatter(distance[:, 0], distance[:, 1], s=0.01, alpha=0.7, c=traj.time)
#plt.scatter(distance[:, 0], distance[:, 1],s=1,alpha=0.3,c=energy)
cbar = plt.colorbar()
cbar.set_label('steps')
plt.xlabel(r'Asp3N-Gly7O [nm]')
plt.xlim(0, 2)
plt.ylabel(r'Asp3N-Thr8O [nm]')
plt.ylim(0, 2)
plt.savefig(topo_num +"_2dist_plot.png")

#contact_featurizer
"""
featurizer=ContactFeaturizer(contacts='all', scheme='closest-heavy', ignore_nonprotein=True)
contact=np.array(featurizer.transform(traj))
print(contact.shape)
contact=np.delete(contact, 0, 0)
contact=np.reshape(contact, (1, -1, 28))
print(contact.shape)
print(contact)
"""
"""
    #        np.save(str(i)+"_skip2_data.npy", data1)

    ###input形式：(100, 100, 30)#
    ###input形式：(500, 1000, 30)#
    
    data=np.exp(-data)
    
    
    
    
    data2=np.concatenate([data1_plot, data1], axis=2)
    print(data2.shape)
    
 
"""