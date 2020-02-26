import json
import glob
import numpy as np
import os

path = "data_state_space_v3/"
out_path = "small_data/"
files = glob.glob(path + "*.npy")  # ワイルドカードが使用可能
train_data_num = 100
test_data_num = 10
train_data = {}
test_data = {}
for filename in files:
    obj = np.load(filename)
    if filename.find("_test.npy") >= 0:
        test_data[filename] = obj
    else:
        train_data[filename] = obj
os.makedirs(out_path, exist_ok=True)
for k, v in train_data.items():
    b = os.path.basename(k)
    print(b, v.shape)
    o = v[:train_data_num]
    np.save(out_path + b, o)

for k, v in test_data.items():
    b = os.path.basename(k)
    print(b, v.shape)
    o = v[:test_data_num]
    np.save(out_path + b, o)
fp = open(path + "pack_selected_info.json")
obj = json.load(fp)
obj["pid_list_train"] = obj["pid_list_train"][:train_data_num]
obj["pid_list_test"] = obj["pid_list_test"][:test_data_num]
fp = open(out_path + "pack_selected_info.json", "w")
json.dump(obj, fp)
