import numpy as np
import joblib
import json
import sys
import os
from matplotlib.colors import LinearSegmentedColormap
import argparse
from plot_input import load_plot_data, get_default_argparser


def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list("custom_cmap", color_list)


###
def draw_heatmap(h1, h2, cmap, attr):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 1, 1)
    plt.imshow(
        h1, aspect="auto", interpolation="none", cmap=cmap, vmin=-1.0, vmax=1.0
    )  #
    plt.gca().xaxis.set_ticks_position("none")
    plt.gca().yaxis.set_ticks_position("none")
    if h2 is not None:
        plt.subplot(2, 1, 2)
        plt.imshow(
            h2, aspect="auto", interpolation="none", cmap=cmap, vmin=-1.0, vmax=1.0
        )  #
        plt.gca().xaxis.set_ticks_position("none")
        plt.gca().yaxis.set_ticks_position("none")


parser = get_default_argparser()
parser.add_argument(
    "--obs_dim", type=int, default=0, help="a dimension of observation for plotting"
)

args = parser.parse_args()
# config

if not args.show:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pylab as plt
else:
    from matplotlib import pylab as plt
data = load_plot_data(args, result_key="save_result_test")

# d=data_info["attr_emit_list"].index("206010")
# print("206010:",d)
d = 0

print(data.result.keys())

# z
if "z_q" in data.result:
    z_q = data.result["z_q"]
    mu_q = data.result["mu_q"]
else:
    z_q = data.result["z_params"][0]
print("z_q.shape=", z_q.shape)
# obs
obs_mu = data.result["obs_params"][0]
# pred_mu=data.result["pred_params"][0]
pred_mu = data.result["obs_pred_params"][0]
if data.mask is not None:
    data.obs[data.mask < 0.1] = np.nan
    obs_mu[data.mask < 0.1] = np.nan
    pred_mu[data.mask < 0.1] = np.nan


def plot_fig(idx):
    d = args.obs_dim
    s = data.steps[idx]
    print("data index:", idx)
    # print("data =",data.info[data.pid_key][idx])
    print("error=", np.nanmean((obs_mu[:, :-1, :] - data.obs[:, 1:, :]) ** 2))
    print("dimension (observation):", d)
    print("steps:", s)
    plt.subplot(2, 1, 2)
    plt.plot(data.obs[idx, :s, d], label="x")
    plt.plot(obs_mu[idx, :s, d], label="recons")
    plt.plot(pred_mu[idx, :s, d], label="pred")
    plt.legend()
    plt.subplot(2, 1, 1)
    if False:
        plt.plot(mu_q[idx, :s, 0], label="dim0")
        plt.plot(mu_q[idx, :s, 1], label="dim1")
    else:
        cmap = generate_cmap(["#0000FF", "#FFFFFF", "#FF0000"])
        h = z_q[idx, :s, :]
        # h=mu_q[idx,:s,:]
        print("upper plot:z_q:", h.shape)
        plt.imshow(
            np.transpose(h),
            aspect="auto",
            interpolation="none",
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
        )  #
        plt.gca().xaxis.set_ticks_position("none")
        plt.gca().yaxis.set_ticks_position("none")
    plt.legend()


if args.mode == "all":
    l = len(data.info[data.pid_key])
    if args.limit_all is not None and l > args.limit_all:
        l = args.limit_all
    for idx in range(l):
        plot_fig(idx)
        name = data.info[data.pid_key][idx]
        out_filename = data.out_dir + "/" + str(idx) + "_" + name + "_plot.png"
        print("[SAVE] :", out_filename)
        plt.savefig(out_filename)
        plt.clf()
else:
    idx = args.index
    plot_fig(idx)
    name = data.info[data.pid_key][idx]
    out_filename = data.out_dir + "/" + str(idx) + "_" + name + "_plot.png"
    print("[SAVE] :", out_filename)
    plt.savefig(out_filename)
    plt.show()
    plt.clf()
