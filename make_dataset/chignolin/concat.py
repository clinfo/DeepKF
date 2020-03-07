#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


list = ["0_31410", "0_31414", "0_31418"]
np_list = []
for name in list:
    np_list.append(name + "_dij.npy")

for i in range(3):
    data = np.load(np_list[i])
    if i == 0:
        data1 = data
    if i >= 1:
        data1 = np.concatenate([data1, data])
print(data1.shape)
np.save("3_traj_dij.npy", data1)
