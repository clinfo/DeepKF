import numpy as np
from matplotlib import pylab as plt
import joblib
from matplotlib import animation
import json
import sys

filename_result="result/test.jbl"
filename_obs="data/pack_data_emit_test.npy"
filename_mask="data/pack_mask_emit_test.npy"
filename_info="data/pack_info.json"
pid_key="pid_list_test"
out_dir="plot_test"

if len(sys.argv)>1:
	fp = open(sys.argv[1], 'r')
	config=json.load(fp)
	filename_result=config["save_result_test"]
	filename_obs=config["data_test_npy"]
	filename_mask=config["mask_test_npy"]
	filename_steps=config["steps_test_npy"]
	if "plot_path" in config:
		out_dir=config["plot_path"]


steps=np.load(filename_steps)
pid_key="pid_list_test"
obj=joblib.load(filename_result)
z_q=obj["z_q"].reshape(obj["mu_q"].shape)
mu_q=obj["mu_q"]
cov_q=obj["cov_q"]
obs_mu=obj["obs_params"][0]
x2=obj["x"]
#
fp = open(filename_info, 'r')
data_info = json.load(fp)
d=data_info["attr_emit_list"].index("206010")
idx=3
l=len(data_info[pid_key])
def plot_mv(idx):
	s=int(steps[idx])
	print(s)
	x=mu_q[idx,:s,0]
	y=mu_q[idx,:s,1]
	fig=plt.figure()
	plt.subplot(2,1,1)
	plt.plot(x[1],y[1],"ro")
	plt.plot(x[1:s],y[1:s],label="mu_q")
	plt.legend()
	line, = plt.plot([], [],"ro", lw=2)

	plt.subplot(2,1,2)
	line2_x, = plt.plot([],[], lw=1,label="x")
	line2_o, = plt.plot([],[], lw=2,label="obs_mu")
	x_window=50
	plt.xlim(0,x_window)
	plt.ylim(-1.5,1.5)
	plt.legend()
	def init():
		line.set_data([], [])
		return line,
	def animate(i):
		if x_window+i<x2.shape[1]:
			px=x[i]
			py=y[i]
			line.set_data([px,px],[py,py])
			line2_x.set_data(range(x_window),x2[idx,i:x_window+i,d])
			line2_o.set_data(range(x_window),obs_mu[idx,i:x_window+i,d])
			return line,line2_x,line2_o
		return None

	anim = animation.FuncAnimation(fig, animate, init_func=init,
			frames=s, interval=500, blit=True)
	return anim

print(data_info[pid_key][idx])
name=data_info[pid_key][idx]
#plt.show()
plt.clf()


import numpy as np
import joblib
import json
import sys
import os
from matplotlib.colors import LinearSegmentedColormap
import argparse
from dmm.plot_input import load_plot_data, get_default_argparser
from matplotlib import pylab as plt

def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list("custom_cmap", color_list)

class PlotFig():

    def draw_heatmap(self,h1, cmap, vmin=-1, vmax=1, relative=True):
        if relative:
            im =plt.imshow(h1, aspect="auto", interpolation="none", cmap=cmap,origin='lower')
            plt.colorbar(im)
        else:
            plt.imshow(h1, aspect="auto", interpolation="none", cmap=cmap,origin='lower',vmin=vmin,vmax=vmax)
        plt.gca().xaxis.set_ticks_position("none")
        plt.gca().yaxis.set_ticks_position("none")

    # line: num x steps
    def draw_line(self,line,label,color,num):
        plt.plot(line[0,:],label=label,color=color)
        for i in range(num-1):
            plt.plot(line[i+1,:],color=color)
        
    def draw_scatter_z(self,x,y,c,alpha=1.0,axis_x=0,axis_y=1):
        plt.scatter(x,y,c=c,alpha=alpha)
        min_x,max_x=self.min_z[axis_x],self.max_z[axis_x]
        offset_x=(max_x-min_x)*0.01
        plt.xlim(min_x-offset_x,max_x+offset_x)
        min_y,max_y=self.min_z[axis_y],self.max_z[axis_y]
        offset_y=(max_y-min_y)*0.01
        plt.ylim(min_y-offset_y,max_y+offset_y)

    def build_data(self,args,data,result_data):
        self.colorlist = ["g", "b", "r", "c", "m", "y", "k", "w"]
        self.cmap_z = generate_cmap(["#0000FF", "#FFFFFF", "#FF0000"])
        self.cmap_x = generate_cmap(["#FFFFFF", "#000000"])
        # z
        self.z_q=None
        self.mu_q=None
        self.sample_z=None
        if "z_q" in result_data:
            # deprecated
            self.z_q = result_data["z_q"]
            if "mu_q" in result_data:
                self.mu_q = result_data["mu_q"]
                print("z: z_q, mu_q")
            else:
                print("z: z_q")
        elif "z_params" in result_data:
            self.z_q = result_data["z_params"][0]
            print("z: z_params")
        elif "z" in result_data:
            # sampling
            self.z_q = np.mean(result_data["z"],axis=0)
            self.sample_z=result_data["z"]
            print("z: z")
        else:
            print("z: nothing")
        if self.z_q is not None:
            print("z: ", self.z_q.shape)
        # recons
        self.obs_mu=None
        self.sample_obs=None
        if "obs_params" in result_data:
            self.obs_mu = result_data["obs_params"][0]
            print("reconstructed obs:", obs_mu.shape)
        elif "mu" in result_data:
            # sampling
            self.obs_mu = np.mean(result_data["mu"],axis=0)
            self.sample_obs=result_data["z"]
            print("reconstructed obs:", self.obs_mu.shape)
        else:
            print("reconse: nothing")
        print("obs:", data.x.shape)
        # pred_mu=result_data["pred_params"][0]
        if "obs_pred_params" in result_data:
            self.pred_mu = result_data["obs_pred_params"][0]
        if data.mask is not None and np.any(data.mask < 0.1):
            data.x[data.mask < 0.1] = np.nan
            if self.obs_mu: self.obs_mu[data.mask < 0.1] = np.nan
            if self.pred_mu: self.pred_mu[data.mask < 0.1] = np.nan

        sample_num_z=10000
        self.all_z = np.reshape(self.z_q[:, :, :],(-1,self.z_q.shape[2]))
        self.max_z=np.max(self.all_z,axis=0)
        self.min_z=np.min(self.all_z,axis=0)
        print(self.max_z)
        print(self.min_z)
        idx=np.random.randint(len(self.all_z), size=sample_num_z)
        self.all_z=self.all_z[idx,:]


    def plot_fig(self, idx, args, data, result_data):
        colorlist=self.colorlist
        colorlist_z=self.colorlist
        cmap_x=self.cmap_x
        cmap_z=self.cmap_z
        d = args.obs_dim
        if data.steps:
            s = data.steps[idx]
        else:
            s=data.x.shape[1]
        print("data index:", idx)
        # print("data =",data.info[data.pid_key][idx])
        if self.obs_mu is not None:
            error=(self.obs_mu[idx, :-1, :] - data.x[idx, 1:, :]) ** 2
            print("error=", np.nanmean(error))
        print("dimension (observation):", d)
        print("steps:", s)

        if args.mode=="data":
            plt.subplot(1, 1, 1)
            self.draw_heatmap(np.transpose(data.x[idx, :s, :]), cmap_x)
            plt.legend()
            plt.title("x: observation")

        elif args.mode=="filter":
            plt.subplot(3,1,1)
            self.draw_line(self.sample_z[:,idx,:s,d],label="dim-"+str(d),color=colorlist[d],num=args.num_particle)
            plt.legend()
            plt.title("z: state space")
            
            plt.subplot(3,1,2)
            plt.plot(data.x[idx,:s,d],label="x",color="b")
            self.draw_line(self.sample_obs[:,idx,:s,d],label="pred",color="g",num=args.obs_num_particle)
            plt.plot(self.obs_mu[idx,:s,d],label="pred_mean",color="r")
            plt.legend()
            plt.title("x: observation")

            plt.subplot(3,1,3)
            e=(self.sample_obs[:,idx,:s,d]-data.x[idx,:s,d])**2
            self.draw_line(e,label="error",color="b",num=args.obs_num_particle)
            plt.legend()
            plt.title("error")
        else:

            plt.subplot(2, 1, 1)
            z = self.z_q[idx, :s, :]
            z_plot_type="scatter"
            if z_plot_type=="line":
                plot_dim=min([len(colorlist_z),z.shape[1]])
                for i in range(plot_dim):
                    plt.plot(z[:, i], label="dim-"+str(i)+"/"+str(z.shape[1]))
            elif z_plot_type=="heatmap":
                # h=mu_q[idx,:s,:]
                print("upper plot:z_q:", z.shape)
                self.draw_heatmap(np.transpose(z), cmap_z)
            else:
                self.draw_scatter_z(self.all_z[:,0],self.all_z[:,1],c="b",alpha=0.1)
                self.draw_scatter_z(z[:,0],z[:,1],c="r",alpha=0.8)
            plt.legend()
            plt.title("z: state space")
            
            x_plot_type="heatmap"
            plt.subplot(2, 1, 2)
            if x_plot_type=="line":
                plt.plot(data.x[idx, :s, d], label="x")
                plt.plot(self.obs_mu[idx, :s, d], label="recons")
                plt.plot(self.pred_mu[idx, :s, d], label="pred")
                plt.legend()
            else:
                self.draw_heatmap(np.transpose(data.x[idx, :s, :]), cmap_x)
            plt.legend()
            plt.title("x: observation")
            def init():
                line.set_data([], [])
                return line,
            def animate(i):
                if x_window+i<x2.shape[1]:
                    px=x[i]
                    py=y[i]
                    line.set_data([px,px],[py,py])
                    line2_x.set_data(range(x_window),x2[idx,i:x_window+i,d])
                    line2_o.set_data(range(x_window),obs_mu[idx,i:x_window+i,d])
                    return line,line2_x,line2_o
                return None

            anim = animation.FuncAnimation(fig, animate, init_func=init,
                    frames=s, interval=500, blit=True)
        return anim

       


def plot_start():
    parser = get_default_argparser()
    parser.add_argument(
        "--obs_dim", type=int, default=0, help="a dimension of observation for plotting"
    )
    parser.add_argument('--num_particle', type=int,
        default=10,
        help='the number of particles (latent variable)')
    parser.add_argument('--obs_num_particle', type=int,
        default=10,
        help='the number of particles (observation)')

    args = parser.parse_args()
    # config
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pylab as plt
    else:
        from matplotlib import pylab as plt
    ##
    print("=============")
    print("... loading")
    # d=data_info["attr_emit_list"].index("206010")
    # print("206010:",d)
    if args.config is None:
        parser.print_help()
        quit()
    data,result_data = load_plot_data(args)
    print("data:",data.keys())
    print("result:",result_data.keys())
    print("=============")
    plotter=AnimeFig()
    plotter.build_data(args,data,result_data)
    if args.index is None:
        l = len(result_data.info[result_data.pid_key])
        if args.limit_all is not None and l > args.limit_all:
            l = args.limit_all
        for idx in range(l):
            anim = plotter.plot_fig(idx,args,data,result_data)
            name = result_data.info[result_data.pid_key][idx]
            out_filename = result_data.out_dir + "/" + str(idx) + "_" + name + "_"+args.mode+".mp4"
            print("[SAVE] :", out_filename)
            anim.save(out_filename, fps=10, extra_args=['-vcodec', 'libx264'])
    else:
        idx = args.index
        plotter.plot_fig(idx,args,data,result_data)
        name = result_data.info[result_data.pid_key][idx]
        out_filename = result_data.out_dir + "/" + str(idx) + "_" + name + "_"+args.mode+".mp4"
        print("[SAVE] :", out_filename)
        anim.save(out_filename, fps=10, extra_args=['-vcodec', 'libx264'])
def main():
    plot_start()
