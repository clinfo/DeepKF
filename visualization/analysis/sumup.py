import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pylab as plt
import joblib
import json
import sys
import argparse
import pickle
import math
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

errors = data.result["error"]

E = []
for i in range(idx):
    e_sum = []
    for d in range(2, dim):
        for s in range(step):
            e_sum.append(float(errors[0, i, s, d] ** 2))
    E.append(math.sqrt(sum(e_sum)))

print(E)
x = list(range(0, 501))

plt.bar(np.array(x[0:500]), np.array(E[0:500]))
plt.ylim(0, 40)
plt.savefig("error_barplot_dim2.png")

# f=open("error_nomask.txt", "wb")
# pickle.dump(E, f)
