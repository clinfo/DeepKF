import numpy as np
import os
import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

"""
info={
        "attr_emit_list": ["item1"],
        "pid_list_train": [],
        "pid_list_test": []
        }
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def lorenz_f(xyz, p=10, r=28, b=8 / 3.0):
    return [
        -p * xyz[0] + p * xyz[1],
        -xyz[0] * xyz[2] + r * xyz[0] - xyz[1],
        xyz[0] * xyz[1] - b * xyz[2],
    ]


def run(x0=1, y0=1, z0=1, step=100000, dt=1e-3):
    """ Loranz attractor (Runge-Kutta method) execution """
    res = [[], [], []]
    xyz = [x0, y0, z0]
    for _ in range(step):
        k_0 = lorenz_f(xyz)
        k_1 = lorenz_f([x + k * dt / 2 for x, k in zip(xyz, k_0)])
        k_2 = lorenz_f([x + k * dt / 2 for x, k in zip(xyz, k_1)])
        k_3 = lorenz_f([x + k * dt for x, k in zip(xyz, k_2)])
        for i in range(3):
            xyz[i] += (k_0[i] + 2 * k_1[i] + 2 * k_2[i] + k_3[i]) * dt / 6.0
            res[i].append(xyz[i])
    return np.array(res)


def plot3d(res):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Lorenz attractor (Runge-Kutta method)")
    ax.plot(res[0], res[1], res[2], color="red", lw=1)
    # plt.show()
    plt.savefig("lorenz_attractor_runge_kutta.png")


def plot_lines(res):
    plt.figure(figsize=(10, 1))
    l = res.shape[1]
    plt.plot(list(range(l)), res[0, :], color="red", lw=1)
    plt.figure(figsize=(10, 1))
    plt.plot(list(range(l)), res[1, :], color="red", lw=1)
    plt.figure(figsize=(10, 1))
    plt.plot(list(range(l)), res[2, :], color="red", lw=1)
    # plt.show()
    plt.savefig("lorenz_attractor_runge_kutta.png")
def get_data(num):
    result=[]
    for _ in range(num):
        res = run(
            np.random.rand(), np.random.rand(), np.random.rand(), step=120000, dt=1e-3
        )
        res = res[:, 20000:]
        n = res.shape[1]
        idx = np.linspace(0, n, num=n / 100, dtype="int", endpoint=False)
        res = res[:, idx]
        result.append(res)

    result = np.array(result)
    result = np.transpose(result, [0, 2, 1])
    result = np.reshape(result, (num, 10, 100, 3))
    result = np.reshape(result, (-1, 100, 3))
    np.random.shuffle(result)
    return result
def add_noise(result,scale=0):
    if scale:
        result += np.random.normal(scale=scale, size=result.shape)
    return result


os.makedirs("data", exist_ok=True)
np.random.seed(10)
print("data/data_train.npy")
print("data/data_test.npy")
"""
result = get_data(50)
np.save("data/data_train.npy",result)
result = get_data(5)
np.save("data/data_test.npy",result)
"""
result_train = get_data(50)
m=np.mean(result_train,axis=(0,1))
s=np.std(result_train,axis=(0,1))
result_train=(result_train-m)/s
result_test = (get_data(5)-m)/s
print(m)
filename="data/mean.npy"
np.save(filename,m)
filename="data/std.npy"
np.save(filename,s)
for sc in range(10):
    x=np.copy(result_train)
    x=add_noise(x,scale=(sc+1)/10.0)
    filename="data/data_train.n"+str(sc+1)+".npy"
    np.save(filename,x)
    print(filename)
    ##
    x=np.copy(result_test)
    x=add_noise(x,scale=(sc+1)/10.0)
    filename="data/data_test.n"+str(sc+1)+".npy"
    np.save(filename,x)
    print(filename)
