import numpy as np
import joblib
import json
import sys
import os
scrdir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(scrdir)
from matplotlib.colors import LinearSegmentedColormap
import argparse
from plot_input import load_plot_data,get_default_argparser
from math import pi



parser=get_default_argparser()
parser.add_argument('--obs_dim', type=int,
                    default=0,
                    help='a dimension of observation for plotting')

args=parser.parse_args()
data=load_plot_data(args, result_key="save_result_test")

d=0

#print(data.result.keys())
#==== save_result_test
#[LOAD]: sample_md2/model/result/test.jbl
#[LOAD]: sample_md2/dataset/ala_traj_all.npy
#dict_keys(['z_s', 'z_params', 'z_pred_params', 'obs_params', 'obs_pred_params', 'config'])



# z
if "z_q" in data.result:
    z_q=data.result["z_q"]
    mu_q=data.result["mu_q"]
else:
    z_q=data.result["z_params"][0]

print("z_q.shape=",z_q.shape)
#z_q.shape= (600, 300, 1)

# obs
obs_mu=data.result["obs_params"][0]
#pred_mu=data.result["pred_params"][0]
pred_mu=data.result["obs_pred_params"][0]

if data.mask is not None:
    data.obs[data.mask<0.1]=np.nan
    obs_mu[data.mask<0.1]=np.nan
    pred_mu[data.mask<0.1]=np.nan

def dim_val(idx):
    d=args.obs_dim
    s=data.steps[idx]
#    print("data index:",idx)
#    print("data =",data.info[data.pid_key][idx])
#    print("error=",np.nanmean((obs_mu[:,:-1,:]-data.obs[:,1:,:])**2))
#    print("dimension (observation):",d)
#    print("steps:",s)

    dim = z_q[idx:idx+1,:s,0]
#    print("dim size:", dim.shape) #(1, 300)

    return dim
    

def pass_count(data):
    dimlist =data.tolist()
    numlist=[]
    for i in range(len(dimlist)):
        dim = float(dimlist[i])
        if dim <= -0.640:
            numlist.append(1)
        elif -0.640 <dim <=-0.630:
            numlist.append(2)
        elif -0.630 < dim:
            numlist.append(3)
#    print(len(numlist))

    pass_list=[]
    for i in range(len(numlist)):
        if i ==0:
            from_to_dur=[]
            from_to_dur.append(numlist[0])
            num=1
        if i >=1:
            if numlist[i] != numlist[i-1]:
                from_to_dur.append(numlist[i])
#                print(num)
                from_to_dur.append(num)
                pass_list.append(from_to_dur)
                from_to_dur=[]
                from_to_dur.append(numlist[i])
                num=1
            if numlist[i] == numlist[i-1]:
                num +=1

    print("total pass:", len(pass_list))


    num12=0
    num13=0
    num21=0
    num23=0
    num31=0
    num32=0
    for g in range(len(pass_list)):
        if pass_list[g][2] >=100:
            if pass_list[g][0]==1 and pass_list[g][1]==2:
                num12 +=1
            elif pass_list[g][0]==1 and pass_list[g][1]==3:
                num13 +=1
            elif pass_list[g][0]==2 and pass_list[g][1]==1:
                num21 +=1
            elif pass_list[g][0]==2 and pass_list[g][1]==3:
                num23 +=1
            elif pass_list[g][0]==3 and pass_list[g][1]==1:
                num31 +=1
            elif pass_list[g][0]==3 and pass_list[g][1]==2:
                num32 +=1
            else:
                print(pass_list[g])

    time12= []
    time13= []
    time21= []
    time23= []
    time31= []
    time32= []
    for g in range(len(pass_list)):
        if pass_list[g][0]==1 and pass_list[g][1]==2:
            time12.append(pass_list[g][2])
        elif pass_list[g][0]==1 and pass_list[g][1]==3:
            time13.append(pass_list[g][2])
        elif pass_list[g][0]==2 and pass_list[g][1]==1:
            time21.append(pass_list[g][2])
        elif pass_list[g][0]==2 and pass_list[g][1]==3:
            time23.append(pass_list[g][2])
        elif pass_list[g][0]==3 and pass_list[g][1]==1:
            time31.append(pass_list[g][2])
        elif pass_list[g][0]==3 and pass_list[g][1]==2:
            time32.append(pass_list[g][2])
        else:
            print(pass_list[g])

    print("1→2:",sum(time12)/len(time12))
    print("1→3:",sum(time13)/len(time13))
    print("2→1:",sum(time21)/len(time21))
    print("2→3:",sum(time23)/len(time23))
#    print(sum(time31)//len(time31))
    print("3→2:",sum(time32)/len(time32))
#    return 12, num13, num21, num23, num31, num32

    print("1→2:",max(time12))
    print("1→3:",max(time13))
    print("2→1:",max(time21))
    print("2→3:",max(time23))
#    print(sum(time31)//len(time31))
    print("3→2:",max(time32))

    return num12, num13, num21, num23, num31, num32



if args.mode=="all":
    l=len(data.info[data.pid_key])
    if args.limit_all is not None and l > args.limit_all:
        l = args.limit_all
    for idx in range(l):
        if idx == 0:
            dim = dim_val(idx)
        if idx >= 1:
            add_dim = dim_val(idx)
            dim = np.concatenate([dim, add_dim])
    print("final dim size:", dim.shape)

    dim_data =dim.reshape(-1)
    num12, num13, num21, num23,num31, num32 = pass_count(dim_data)
    print(num12, num13, num21, num23, num31, num32)


    
else:
    pass
