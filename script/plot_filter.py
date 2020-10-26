import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import joblib

def plot_filter(filter_jbl,filter_image):
    data = joblib.load(filter_jbl)
    z_traj = data["z"][0][0]
    fig = plt.figure(figsize=(6.0,6.0))
    plt.title("latent space")
    plt.xlim(-5.0,5.0)
    plt.ylim(-5.0,5.0)
    #i=0
    artists = []
    
    for i in range(100):
        image = plt.plot(z_traj[i][0],z_traj[i][1],marker="o",color="blue")
        itr = [plt.text(3.0,5.0,"iteration number : "+str(i))]
        artists.append(image+itr)
    ani = animation.ArtistAnimation(fig,artists)
    ani.save("sample_plot_filter/sample_plot_filter.gif")
    #print(z_traj)

    #traj = data["mu"][0][5]
    #print(traj)    
    #plt.figure(figsize=(16.0,6.0))
    #plt.plot(traj,linestyle="None",marker="o")
    #plt.savefig(filter_image)

if __name__ == '__main__':
    args = sys.argv
    plot_filter(args[1],args[2])