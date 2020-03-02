#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import mdtraj as md
import numpy
from pylab import *
from math import pi
import sys

def get_angles(fname1):
    traj = md.load_dcd(fname1, top = 'ala2.pdb')
    atoms, bonds = traj.topology.to_dataframe()
    psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]
    arr_angles = md.compute_dihedrals(traj, [phi_indices, psi_indices])
    return traj, arr_angles

if __name__ == "__main__":
    argvs = sys.argv
    fname1 = str(argvs[1])
    oname = str(argvs[2])
    traj, angles = get_angles(fname1)

    plt.figure()
    plt.title('Dihedral Map: Alanine dipeptide')
    plt.scatter(angles[:, 0], angles[:, 1], marker='x', c=traj.time)
    cbar = plt.colorbar()
    cbar.set_label('Time [ps]')
    plt.xlabel(r'$\Phi$ Angle [radians]')
    plt.xlim(-pi, pi)
    plt.ylabel(r'$\Psi$ Angle [radians]')
    plt.ylim(-pi, pi)
    plt.savefig(oname)
