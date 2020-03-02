#! /usr/bin/env python

import numpy as np
import joblib
import json
import os
import sys
from matplotlib.colors import LinearSegmentedColormap
from plot_input import load_plot_data,get_default_argparser
from math import pi
import argparse
from matplotlib.gridspec import GridSpec,GridSpecFromSubplotSpec

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt


filename_info="data/pack_info.json"
filename_result="sim.jbl"
pid_key="pid_list_test"
out_dir="plot_test"

parser=get_default_argparser()
args=parser.parse_args()

if args.config is not None:
    fp = open(args.config, 'r')
    config=json.load(fp)
    if "plot_path" in config:
        out_dir=config["plot_path"]
        filename_result=config["simulation_path"]+"/field.jbl"


print("[LOAD] ",filename_result)
obj=joblib.load(filename_result)
data_z=obj["z"]
data_gz=obj["gz"]
print("shape z:",data_z.shape)
print("shape grad. z",data_gz.shape)

#画像全体のサイズ指定
figure =plt.figure(figsize=(15, 8))
gs_master = GridSpec(nrows=3, ncols=4, height_ratios=[0.1, 1, 0.2], width_ratios=[1, 0.2, 0.2, 1.1])


#矢印の図
X=data_z[:,0]
Y=0
U=data_gz[:,0]
V=0
R=np.sqrt(U**2+V**2)
print(X)
print(U)
#plt.axes([0.025, 0.025, 0.95, 0.95])

gs_1 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[0, 0])
axis_1 = figure.add_subplot(gs_1[:,:])
axis_1.quiver(X, Y, U, V, R, alpha=.5)
#plt.quiver(X, Y, U, V, edgecolor='k', facecolor='None', linewidth=.5)
r=2.0#20
axis_1.set_xlim(-1, 1)


#積分の図
x=X.tolist()
u=U.tolist()

y=[]
h=0
for i in range(30):
    h += float(u[i])
    y.append(round(h, 2))

y2=[]
y2+=[max(y)-y[i] for i in range(30)]
y2_arr=np.array(y2).reshape(30, 1)

gs_2 = GridSpecFromSubplotSpec(nrows=1, ncols=1, subplot_spec=gs_master[1, 0])
axis_2 = figure.add_subplot(gs_2[:,:])
axis_2.plot(x, y2_arr)
axis_2.set_xlim(-1, 1)
axis_2.set_ylim(np.amin(y2_arr), 1)


#　x軸の生成
x2 = np.linspace(-r, r, len(x))
#print(x2)
#print(y2)
#　フィッティング
a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, b = np.polyfit(x2, y2_arr, 16)
# フィッティング曲線
#fh = a1 * x2**2 + a2 * x2 +b
fh = a1 * x2**16 + a2 * x2**15 + a3 * x2**14 + a4 * x2**13 + a5 *x2**12 + a6 *x2**11 + a7 *x2**10 +a8 *x2**9 +a9*x2**8 +a10 *x2**7 +a11 *x2**6 +a12 *x2**5 +a13 *x2**4 +a14 *x2**3 +a15 *x2**2+a16 *x2 + b

# フィッティング曲線のプロット
axis_2.plot(x2, fh, label="fh", color="r")
#print(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, b)

def kansuu(x2):
    y2 = a1 * x2**16 + a2 * x2**15 + a3 * x2**14 + a4 * x2**13 + a5 *x2**12 + a6 *x2**11 + a7 *x2**10 +a8 *x2**9 +a9*x2**8 +a10 *x2**7 +a11 *x2**6 +a12 *x2**5 +a13 *x2**4 +a14 *x2**3 +a15 *x2**2+a16 *x2 + b
    return y2
print(kansuu(1))

def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


def draw_heatmap(h1,h2,cmap,attr):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 1, 1)
    plt.imshow(h1, aspect='auto', interpolation='none',
                cmap=cmap, vmin=-1.0, vmax=1.0)#
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    if h2 is not None:
        plt.subplot(2, 1, 2)
        plt.imshow(h2, aspect='auto', interpolation='none',
                    cmap=cmap, vmin=-1.0, vmax=1.0)#
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')


data=load_plot_data(args,result_key="save_result_test")
d=0
print(data.result.keys())

# z
if "z_q" in data.result:
    z_q=data.result["z_q"]
    mu_q=data.result["mu_q"]
else:
    z_q=data.result["z_params"][0]

print("z_q.shape=",z_q.shape)

# obs
obs_mu=data.result["obs_params"][0]
pred_mu=data.result["obs_pred_params"][0]

if data.mask is not None:
    data.obs[data.mask<0.1]=np.nan
    obs_mu[data.mask<0.1]=np.nan
    pred_mu[data.mask<0.1]=np.nan


def plot_fig(idx):
    d=1
    s=data.steps[idx]
    print("data index:",idx)
    print("error=",np.nanmean((obs_mu[:,:-1,:]-data.obs[:,1:,:])**2))
    print("dimension (observation):",d)
    print("steps:",s)
    
    cmap = generate_cmap(['#0000FF','#08FF00','#FFE900', '#FF0000'])
#
    gs_3 = GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs_master[:2,3])
    axis_3 = figure.add_subplot(gs_3[:,:])
    pc=axis_3.scatter(-np.log(data.obs[:idx+1,:s,0]), -np.log(data.obs[:idx+1,:s,1]), label="x", s=5, alpha=0.3, c= kansuu(z_q[:idx+1,:s,0]), cmap =cmap)
    axis_3.set_xlabel(r'Asp3N-Gly7O [nm]')
    axis_3.set_xlim(0, 2)
    axis_3.set_ylabel(r'Asp3N-Thr8O [nm]')
    axis_3.set_ylim(0, 2)
    
    gs_4 = GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_master[1,1])
    axis_4 = figure.add_subplot(gs_4[:,:])
    cbar = plt.colorbar(pc, cax = axis_4)
    cbar.set_label('pot')
    cbar.set_clim(0, 1)


if args.mode=="all":
    idx=499
    plot_fig(idx)
    out_filename="test_plot.png"
    print("[SAVE] :",out_filename)
    plt.savefig(out_filename)
    plt.clf()
