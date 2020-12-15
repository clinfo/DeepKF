import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import joblib
import numpy as np

def plot_filter(filter_jbl,filter_image):
    data = joblib.load(filter_jbl)
    z_traj = data["z"][0][0]
    x_traj = data["mu"][0][0]
    fig = plt.figure(figsize=(24.0,24.0))
    ax1 = fig.add_subplot(2,2,1)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    ax1.set_title("latent space")
    ax1.set_xlim(-5.0,5.0)
    ax1.set_ylim(-5.0,5.0)
    ax3.set_title("observable space")
    ax3.set_xlim(0,1000)
    ax3.set_ylim(-1.0,5.0)
    ax4.set_title("observable space")
    ax4.set_xlim(-1.0,5.0)
    ax4.set_ylim(0.0,2.2)
    grid3 = range(1000)
    ax3.scatter(grid3,x_traj[0:1000,0],vmin=-1.0,vmax=5.0,c=x_traj[:1000,0],cmap="cool")
    ax4.hist(x_traj[:,0],bins=200,density=True,color="black")
    ax1.scatter(z_traj[:1000,0],z_traj[:1000,1],vmin=-1.0,vmax=5.0,c=x_traj[:1000,0],cmap="cool")
    fig.savefig(filter_image)

if __name__ == '__main__':
    args = sys.argv
    plot_filter(args[1],args[2])