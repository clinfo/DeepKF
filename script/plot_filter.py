import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import joblib
import numpy as np

# from data_generator.py
def asymmetric_double_well_energy(x):
    r"""computes the potential energy at point x"""
    _x = x - 2.0
    return 2.0 * _x - 6.0 * _x**2 + _x**4

def plot_filter(filter_jbl,filter_image):
    data = joblib.load(filter_jbl)
    z_traj = data["z"][0][0]
    x_traj = data["mu"][0][0]
    fig = plt.figure(figsize=(24.0,24.0))
    ax1 = fig.add_subplot(2,2,1)
    ax2 = fig.add_subplot(2,2,2)
    ax3 = fig.add_subplot(2,2,3)
    ax4 = fig.add_subplot(2,2,4)
    ax1.set_title("latent space")
    ax1.set_xlim(-5.0,5.0)
    ax1.set_ylim(-5.0,5.0)
    ax2.set_title("observable space")
    ax2.set_xlim(-1.0,5.0)
    ax2.set_ylim(-20,10)
    ax3.set_title("observable space")
    ax3.set_xlim(0,100)
    ax3.set_ylim(-1.0,5.0)
    ax4.set_title("observable space")
    ax4.set_xlim(-1.0,5.0)
    ax4.set_ylim(0.0,1.0)
    grid = np.linspace(start=-1.0,stop=5.0,num=1000)
    artists = []
    ax2.plot(grid,asymmetric_double_well_energy(grid),color="black")
    grid3 = range(100)
    ax3.plot(grid3,x_traj[0:100,0],marker="o",color="black",linestyle="None")
    #ax3.axvline(x=3,ymin=-1.0,ymax=5.0)
    ax4.hist(x_traj[:,0],bins=20,normed=True)
    for i in range(100):
        x = x_traj[i,0]
        z_image = ax1.scatter(z_traj[:i,0],z_traj[:i,1],
        vmin=-1.0,vmax=5.0,c=x_traj[:i,0],cmap="cool")
        x_image = ax2.plot(x,asymmetric_double_well_energy(x),marker="o",color="red",linestyle='None')
        #vline = ax3.axvline(x=i,ymin=-1.0,ymax=5.0)
        x_time = ax3.plot(i,x,marker="o",color="red",linestyle="None")
        itr = [ax1.text(2.0,5.0,"step : "+str(i))]
        #artists.append([z_image]+x_image+itr)
        
        artists.append([z_image]+x_image+x_time+itr)
        #artists.append([z_image]+itr)
    fig.colorbar(z_image,ax=ax1)
    ani = animation.ArtistAnimation(fig,artists)
    ani.save("sample_plot_filter/sample_plot_filter.gif")




if __name__ == '__main__':
    args = sys.argv
    plot_filter(args[1],args[2])