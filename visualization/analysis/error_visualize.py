#!/usr/bin/env python
# coding: utf-8

import pickle
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import numpy as np
import math

f=open('error_nomask.txt', 'rb')
error =pickle.load(f)

print(len(error))

x=[]
data=[]
for i in range(len(error)):
    data.append(math.sqrt(sum(error[i])))
    x.append(i)

print(len(data))
print(len(x))

#誤差合計を棒グラフで出力
plt.bar(np.array(x[0:500]), np.array(data[0:500]))
plt.savefig("error_barplot.png")

data_good=[]
data_bad=[]
idx_good=[]
idx_bad=[]
for h in range(len(data)):
    if data[h]<1000:
        data_good.append(data[h])
        idx_good.append(h)
    else:
        data_bad.append(data[h])
        idx_bad.append(h)

print(len(data_good))
print(len(data_bad))

print(idx_good)



d=open('plotdata_x.txt', 'rb')
x =pickle.load(d)

j=open('plotdata_y.txt', 'rb')
y =pickle.load(j)

#print(len(plotdata))

for n in range(len(idx_good)):
    plt.scatter(-np.log(x[idx_good[n]][:]), -np.log(y[idx_good[n]][:]), label="x", s=5, alpha=0.3, c="r")
    plt.xlabel(r'Asp3N-Gly7O [nm]')
    plt.xlim(0, 2)
    plt.ylabel(r'Asp3N-Thr8O [nm]')
    plt.ylim(0, 2)


##for m in range(len(idx_bad)):
##    plt.scatter(-np.log(plotdata[idx_bad[m]][0]), -np.log(plotdata[idx_bad[m]][1]), label="x", s=5, alpha=0.3, c="b")
##    plt.xlabel(r'Asp3N-Gly7O [nm]')
##    plt.xlim(0, 2)
##    plt.ylabel(r'Asp3N-Thr8O [nm]')
##    plt.ylim(0, 2)
##
plt.savefig("check_good.png")



#c=open('error_mask.txt', 'rb')
#maskdata =pickle.load(c)
#
#print(len(maskdata[0]))
