import mdtraj as md

def get_angles(fname1):
    """
    Parameter
    ---------
    fname1 : filename of the trajectory data file (.xtc or .dcd)
    """
    if fname1[-4:] == '.dcd':      
        traj = md.load_dcd(fname1, top = 'ala2.pdb')
    elif fname1[-4:] == '.xtc':
        traj = md.load_xtc(fname1, top = 'ala2.pdb')
    else :
        return None
    r"二面角を計算するために, 原子のindexのリストを作っている"
    r"\psi : 7(N ALA),9(CA ALA),15,17(N NME)"
    r"\phi : 5(C ACE),7(N ALA),9(CA ALA),15(C ALA)"
    r"indexはトポロジカルデータから1ずつずれることに注意"
    psi_indices, phi_indices = [6, 8, 14, 16], [4, 6, 8, 14]
    arr_angles = md.compute_dihedrals(traj, [phi_indices, psi_indices])
    return arr_angles
