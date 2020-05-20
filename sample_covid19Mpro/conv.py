import glob
import mdtraj as md
import numpy as np
#topology = md..topology
#print(topology)

#table, bonds = topology.to_dataframe()

N=10
target_index=None
all_data=[]
top='Traj1/protein_conf.gro'
output="out_every1ns_chunk10.npy"
for traj_filename in glob.glob("Traj1/protein_snap_every1ns_*.xtc"):
    print("[LOAD]",traj_filename)
    for traj in md.iterload(traj_filename,top=top,chunk=N):
        print(traj.xyz.shape)
        #center=np.mean(traj.xyz,axis=1)
        #print ('x: %s\t y: %s\t z: %s' % (center[0,0],center[0,1],center[0,2]))
        #md.compute_dihedrals()

        if target_index is None:
            #print(traj.topology.atom(0))
            table, bonds = traj.topology.to_dataframe()
            df=table[table["element"]!="H"]
            idx=list(df.index)
            print("heavy atom:",len(idx))
            target_index=idx
        data=traj.xyz[:,target_index,:]
        n=data.shape[0]
        all_data.append(data.reshape((n,-1)))

all_data=np.array(all_data)   
print("[SAVE]",output)
print(all_data.shape)
np.save(output,all_data)
