import numpy as np
import matplotlib.pyplot as plt
import data_generator

def make_data_train():
    datapoints = int(49999)
    traj = data_generator.get_asymmetric_double_well_data(datapoints)
    data_train_npy = np.expand_dims(traj, 1)
    fig = plt.figure(figsize=(24.0,24.0))
    plt.title("probability of existence",fontsize=48)
    plt.xlim(-1.0,5.0)
    plt.ylim(0.0,1.1)
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    plt.hist(data_train_npy[:,0],bins=200,density=True,color="black")
    fig.savefig("data_hist.png")
    data_train_npy = np.reshape(data_train_npy, (-1, 100, 1))
    np.save('data_train.npy', data_train_npy)
    print("data_train")
    print(data_train_npy.shape)
    with open('data_train.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        print(data_train_npy,file=f)

def make_data_test():
    datapoints = int(499)
    traj = data_generator.get_asymmetric_double_well_data(datapoints)
    data_test_npy = np.expand_dims(traj, 1)
    data_test_npy = np.reshape(data_test_npy, (-1, 100, 1))    
    np.save('data_test.npy', data_test_npy)
    print("data_test")
    print(data_test_npy.shape)
    with open('data_test.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        print(data_test_npy,file=f)
    return data_test_npy


def make_data_gen(data_test_npy):
    data_gen_npy=np.pad(data_test_npy,[(0,0),(0,4900),(0,0)],'constant')
    np.save('data_gen.npy', data_gen_npy)
    with open('data_gen.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        print(data_gen_npy,file=f)

def make_mask_gen():
    mask_gen_npy = np.array([[[1]]*100+[[0]]*4900]*5)   
    np.save('mask_gen.npy', mask_gen_npy)
    print("mask_gen")
    print(mask_gen_npy.shape)
    with open('mask_gen.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        print(mask_gen_npy,file=f)

np.random.seed(0)
make_data_train()
data_test_npy=make_data_test()
make_data_gen(data_test_npy)
make_mask_gen()