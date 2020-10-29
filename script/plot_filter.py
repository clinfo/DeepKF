import sys
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import joblib

def plot_filter(filter_jbl,filter_image):
    data = joblib.load(filter_jbl)
    z_traj = data["z"][0][0]
    x_traj = data["mu"][0][0]
    fig = plt.figure(figsize=(6.0,6.0))
    ax1 = fig.add_subplot(1,1,1)
    ax2 = fig.add_subplot(2,1,2)
    ax1.set_title("latent space")
    ax1.set_xlim(-5.0,5.0)
    ax1.set_ylim(-5.0,5.0)
    ax2.set_title("observable space")
    ax2.set_xlim(-1.0,5.0)
    ax2.set_ylim(-0.9,5.0)
    #i=0
    artists = []
    colors=[]
    for i in range(100):
        x = x_traj[:i,0]
        print("***")
        print(x)
        
        z_image = ax1.scatter(z_traj[:i,0],z_traj[:i,1],
        vmin=-1.0,vmax=5.0,c=x_traj[:i,0],cmap="cool")
        
        x_image = ax2.plot(x,0,marker="o",color="blue",linestyle='None')
        itr = [ax1.text(2.0,5.0,"step : "+str(i))]
        artists.append(z_image+x_image+itr)
        #artists.append([z_image]+itr)
    fig.colorbar(z_image)
    ani = animation.ArtistAnimation(fig,artists)
    ani.save("sample_plot_filter/sample_plot_filter.gif")


    #traj = data["mu"][0][5]
    #print(traj)    
    #plt.figure(figsize=(16.0,6.0))
    #plt.plot(traj,linestyle="None",marker="o")
    #plt.savefig(filter_image)

if __name__ == '__main__':
    args = sys.argv
    plot_filter(args[1],args[2])