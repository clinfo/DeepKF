import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import joblib
import numpy as np

def plot_infer(infer_jbl,infer_image):
    data = joblib.load(infer_jbl)
    fig = plt.figure(figsize=(24.0,24.0))
    plt.title("latent space",fontsize=48)
    plt.xlim(-5.0,5.0)
    plt.ylim(-5.0,5.0)
    plt.xlabel("dim-0/1", size = 48, weight = "light")
    plt.ylabel("dim-0/2", size = 48, weight = "light")
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    for i in range(100):
        z_traj = data["z_s"][i]
        x_traj = data["obs_params"][0][i]  
        plt.scatter(z_traj[:100,0],z_traj[:100,1],vmin=-1.0,vmax=5.0,c=x_traj[:100,0],cmap="cool")
    fig.savefig(infer_image)

if __name__ == '__main__':
    args = sys.argv
    plot_infer(args[1],args[2])