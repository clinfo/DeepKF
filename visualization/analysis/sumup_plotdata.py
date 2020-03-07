import numpy as np
import joblib
import json
import sys
import argparse
import pickle
from plot_input import load_plot_data, get_default_argparser

parser = get_default_argparser()
parser.add_argument(
    "--num_dim", type=int, default=1, help="the number of dim. (latent variables)"
)
parser.add_argument(
    "--num_particle",
    type=int,
    default=10,
    help="the number of particles (latent variable)",
)
parser.add_argument(
    "--obs_dim", type=int, default=0, help="a dimension of observation for plotting"
)
parser.add_argument(
    "--obs_num_particle",
    type=int,
    default=10,
    help="the number of particles (observation)",
)

args = parser.parse_args()
# config

# データ読み込み
data = load_plot_data(args, result_key="save_result_filter")

idx = args.index
if data.mask is not None:
    data.obs[data.mask < 0.1] = np.nan

# x=obj["x"]

if args.mode == "all":
    idx = len(data.info[data.pid_key])
    if args.limit_all is not None and idx > args.limit_all:
        idx = args.limit_all

# z:潜在空間の値
z = data.result["z"]
# print("z:",z[0,idx,:,0])
# print(z.shape)

# mu:presの値
mu = data.result["mu"]
# step数
step = data.steps[idx - 1]
# 特徴量の次元
dim = args.obs_dim

print("data index:", idx)  # 5
print("dimension (observation):", dim)
print("steps:", step)


plot_x = []
plot_y = []
for i in range(idx):
    D_x = []
    D_y = []
    for s in range(step):
        D_x.append(data.obs[i, s, 0])
        D_y.append(data.obs[i, s, 1])
    plot_x.append(D_x)
    plot_y.append(D_y)


f = open("plotdata_x.txt", "wb")
g = open("plotdata_y.txt", "wb")
pickle.dump(plot_x, f)
pickle.dump(plot_y, g)
