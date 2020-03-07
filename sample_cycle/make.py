import numpy as np
import os
import json
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

info = {"attr_emit_list": ["item1"], "pid_list_train": [], "pid_list_test": []}
count = 0
cycle1 = 5
cycle2 = 5
T = 100
N = 100
data_train = []
label_train = []
for i in range(N):
    x2 = np.random.normal(0, 1, T)
    a = np.random.randint(50)
    cnt = 0
    s = 0
    l = []
    for t in range(T):
        if t < a or a + 50 < t:  # state=1
            if cnt >= cycle1:
                cnt = 0
                s = 1 - s
        else:  # state=2
            if cnt >= cycle2:
                cnt = 0
                s = 1 - s
        if s == 1:
            x2[t] += 10
        cnt += 1
        l.append(s)
    label_train.append(l)
    x2 = x2.reshape(-1, 1)
    data_train.append(x2)
    info["pid_list_train"].append("data" + str(count))
    count += 1
filename = "data_train.npy"
np.save(filename, np.array(data_train))

N = 20
data_test = []
label_test = []
for i in range(N):
    x2 = np.random.normal(0, 1, T)
    a = np.random.randint(50)
    cnt = 0
    s = 0
    l = []
    for t in range(T):
        if t < a or a + 50 < t:  # state=1
            if cnt >= cycle1:
                cnt = 0
                s = 1 - s
        else:  # state=2
            if cnt >= cycle2:
                cnt = 0
                s = 1 - s
        if s == 1:
            x2[t] += 10
        cnt += 1
        l.append(s)
    label_test.append(l)
    x2 = x2.reshape(-1, 1)
    data_test.append(x2)
    info["pid_list_test"].append("data" + str(count))
    count += 1

data_train = np.array(data_train)
data_test = np.array(data_test)
print("data_train:", data_train.shape)
print("data_test:", data_test.shape)

filename = "data_train.npy"
np.save(filename, np.array(data_train))
print("[save]", filename)

filename = "data_test.npy"
data_test[:, 30:70, :] = 0
np.save(filename, np.array(data_test))
print("[SAVE]", filename)

filename = "mask_train.npy"
mask_train = np.ones_like(data_train)
# mask_train[:,30:70,:]=0
np.save(filename, np.array(mask_train))
print("[SAVE]", filename)

filename = "mask_test.npy"
mask_test = np.ones_like(data_test)
mask_test[:, 30:70, :] = 0
np.save(filename, np.array(mask_test))
print("[SAVE]", filename)


filename = "label_train.npy"
np.save(filename, np.array(label_train))
print("[SAVE]", filename)
filename = "label_test.npy"
np.save(filename, np.array(label_test))
print("[SAVE]", filename)

filename = "data_all.npy"
np.save(filename, np.concatenate([data_train, data_test], axis=0))
print("[SAVE]", filename)


fp = open("info.json", "w")
json.dump(info, fp)
print("[SAVE]", "info.json")
# obj={"0":"pos","1":"neg"}
# fp=open("synth.label.json","w")
# json.dump(obj,fp)
