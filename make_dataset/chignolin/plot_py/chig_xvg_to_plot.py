#! /usr/bin/env python

import sys
import numpy as np
from matplotlib import pyplot as plt

def get_ene_xy(fname):
    x=[]
    y=[]
    f= open(fname, "r")
    lines= f.readlines()
    for i in range(len(lines)):
        if i>=18:
            line =lines[i][:-1].split()
            if len(line)==2:
                x.append(float(line[0]))
                y.append(float(line[1]))
    return x,y

def ave(poss):   
    print(np.array(poss).shape)
    poss_ave=[]
    poss_ave = np.average(np.array(poss), axis=0).tolist()
    return poss_ave

if __name__ == "__main__":
    top_num_list=["0_31410", "0_31414", "0_31418"]
    for top_num in top_num_list:
        x, y = get_ene_xy(top_num+"/protein_gpu/equil_n1/rmsd.xvg")
        oname=top_num+"_rmsd.png"
        plt.figure()
        plt.xlabel(r'Time [ns]')
        plt.ylabel(r'RMSD [nm]')
        plt.plot(x, y)
        plt.savefig(oname)

