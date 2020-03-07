import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

os.makedirs("data",exist_ok=True)
np.random.seed(10)

filename="config_base.json"
o=json.load(open(filename))
print(o)
for j in range(10):
    i=j+1
    o['data_train_npy']='data/data_train.n'+str(i)+'.npy'
    o['data_test_npy']='data/data_test.n'+str(i)+'.npy'
    path="experiments/result_base_"+str(i)+"/"
    o["result_path"]=path
    os.makedirs(path,exist_ok=True)

    cfg_path="experiments/config/"
    os.makedirs(cfg_path,exist_ok=True)
    filename=cfg_path+"/config_base"+str(i)+".json"
    print(filename)
    f = open(filename, "w")
    json.dump(o, f)

filename="config_pot.json"
o=json.load(open(filename))
print(o)
for j in range(10):
    i=j+1
    o['data_train_npy']='data/data_train.n'+str(i)+'.npy'
    o['data_test_npy']='data/data_test.n'+str(i)+'.npy'
    path="experiments/result_pot_"+str(i)+"/"
    o["result_path"]=path
    os.makedirs(path,exist_ok=True)

    cfg_path="experiments/config/"
    os.makedirs(cfg_path,exist_ok=True)
    filename=cfg_path+"/config_pot"+str(i)+".json"
    print(filename)
    f = open(filename, "w")
    json.dump(o, f)


