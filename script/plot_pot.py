import numpy as np
import joblib
import json
import sys

if len(sys.argv) > 2 and sys.argv[2] == "all":
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pylab as plt
else:
    from matplotlib import pylab as plt


from matplotlib.colors import LinearSegmentedColormap


def generate_cmap(colors):
    """自分で定義したカラーマップを返す"""
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list("custom_cmap", color_list)


filename_info = "data/pack_info.json"
filename_result = "sim.jbl"
pid_key = "pid_list_test"
out_dir = "plot_test"

if len(sys.argv) > 1:
    fp = open(sys.argv[1], "r")
    config = json.load(fp)
    if "plot_path" in config:
        out_dir = config["plot_path"]
        filename_result = config["simulation_path"] + "/potential.jbl"


print("[LOAD] ", filename_result)
obj = joblib.load(filename_result)
data_z = obj["z"]
data_gz = obj["pot"]
print(data_z.shape)
print(data_gz.shape)
#
# fp = open(filename_info, 'r')
# data_info = json.load(fp)
# d=data_info["attr_emit_list"].index("206010")

X = data_z[:, 0]
Y = data_z[:, 1]
U = data_gz[:]
plt.axes([0.025, 0.025, 0.95, 0.95])
cm = generate_cmap(["mediumblue", "limegreen", "orangered"])
# plt.quiver(X, Y, U, U, R, alpha=.5)
# plt.quiver(X, Y, U, U, edgecolor='k', facecolor='None', linewidth=.5)
im = plt.scatter(X, Y, c=U, linewidths=0, alpha=0.8, cmap=cm)
im.set_clim(0, 1.0)
plt.colorbar(im)

r = 3.0  # 20
plt.xlim(-r, r)
# plt.xticks(())
plt.ylim(-r, r)
# plt.yticks(())

out_filename = out_dir + "/pot.png"
print("[SAVE] :", out_filename)
plt.savefig(out_filename)

plt.show()
plt.clf()
