import numpy as np
import matplotlib.pyplot as plt
import data_generator

def make_data_train():
    datapoints = int(49999)
    traj = data_generator.get_asymmetric_double_well_data(datapoints)
    data_train_npy = np.expand_dims(traj, 1)
    data_train_npy = np.reshape(data_train_npy, (-1, 100, 1))
    np.save('data_train.npy', data_train_npy)
    print("data_train")
    print(data_train_npy.shape)
    with open('data_train.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        print(data_train_npy,file=f)

def make_data_test():
    datapoints = int(49999)
    traj = data_generator.get_asymmetric_double_well_data(datapoints)
    data_test_npy = np.expand_dims(traj, 1)
    data_test_npy = np.reshape(data_test_npy, (-1, 100, 1))    
    np.save('data_test.npy', data_test_npy)
    print("data_test")
    print(data_test_npy.shape)
    with open('data_test.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        print(data_test_npy,file=f)


def make_zeros_gen():
    zeros_gen_npy = np.array([[[0]]*100]*5)   
    np.save('zeros_gen.npy', zeros_gen_npy)
    print("zeros_gen")
    print(zeros_gen_npy.shape)
    with open('zeros_gen.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        print(zeros_gen_npy,file=f)

def make_mask_gen():
    mask_gen_npy = np.array([[[0]]*100]*5)   
    np.save('mask_gen.npy', mask_gen_npy)
    print("mask_gen")
    print(mask_gen_npy.shape)
    with open('mask_gen.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        print(mask_gen_npy,file=f)

np.random.seed(0)
make_data_train()
make_data_test()
make_zeros_gen()
make_mask_gen()