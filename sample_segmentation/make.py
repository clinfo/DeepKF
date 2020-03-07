import numpy as np
import os
import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

D = 10
info = {
    "attr_emit_list": ["item" + str(i) for i in range(D)],
    "pid_list_train": [],
    "pid_list_test": [],
}
count = 0
data_train = []
# index=[1,2,3,4,3,2,1]
index_list = [[1, 2, 3, 4, 3, 2, 1], [6, 5, 4, 3, 4, 5, 6]]
for i in range(100):
    x2 = np.random.normal(0, 1, D * 100)
    x2 = x2.reshape(-1, D)
    start = np.random.randint(50)
    interval = np.random.randint(10)
    k = np.random.randint(len(index_list))
    index = index_list[k]
    l = len(index)
    for i, j in enumerate(index):
        x2[start + (l + interval) * 0 + i, j] += 10
        x2[start + (l + interval) * 1 + i, j] += 10
        x2[start + (l + interval) * 2 + i, j] += 10
        x2[start + (l + interval) * 0 + i, j + 1] += 10
        x2[start + (l + interval) * 1 + i, j + 1] += 10
        x2[start + (l + interval) * 2 + i, j + 1] += 10
    data_train.append(x2)
    info["pid_list_train"].append("data" + str(count))
    count += 1
filename = "data_train.npy"
np.save(filename, np.array(data_train))

data_test = []
for i in range(20):
    x2 = np.random.normal(0, 1, D * 100)
    x2 = x2.reshape(-1, D)
    start = np.random.randint(20)
    interval = np.random.randint(10)
    k = np.random.randint(len(index_list))
    index = index_list[k]
    l = len(index)
    for i, j in enumerate(index):
        x2[start + (l + interval) * 0 + i, j] += 10
        x2[start + (l + interval) * 1 + i, j] += 10
        x2[start + (l + interval) * 2 + i, j] += 10
        x2[start + (l + interval) * 0 + i, j + 1] += 10
        x2[start + (l + interval) * 1 + i, j + 1] += 10
        x2[start + (l + interval) * 2 + i, j + 1] += 10
    #
    start = 40 + np.random.randint(20)
    interval = np.random.randint(10)
    k = np.random.randint(len(index_list))
    index = index_list[k]
    l = len(index)
    for i, j in enumerate(index):
        x2[start + (l + interval) * 0 + i, j] += 10
        x2[start + (l + interval) * 1 + i, j] += 10
        x2[start + (l + interval) * 2 + i, j] += 10
        x2[start + (l + interval) * 0 + i, j + 1] += 10
        x2[start + (l + interval) * 1 + i, j + 1] += 10
        x2[start + (l + interval) * 2 + i, j + 1] += 10
    #
    #
    data_test.append(x2)
    info["pid_list_test"].append("data" + str(count))
    count += 1

filename = "data_train.npy"
np.save(filename, np.array(data_train))
print("[SAVE]", filename)

filename = "data_test.npy"
np.save(filename, np.array(data_test))
print("[SAVE]", filename)

filename = "data_all.npy"
np.save(filename, np.array(data_train + data_test))
print("[SAVE]", filename)

fp = open("info.json", "w")
json.dump(info, fp)
print("[SAVE]", "info.json")
# obj={"0":"pos","1":"neg"}
# fp=open("synth.label.json","w")
# json.dump(obj,fp)
