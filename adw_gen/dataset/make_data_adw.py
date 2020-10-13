import numpy as np
import matplotlib.pyplot as plt
import data_generator

def make_data_train():
    datapoints = int(100000)
    traj = data_generator.get_asymmetric_double_well_data(datapoints)
    data_train_npy = np.expand_dims(traj[0:50000], 1)
    data_train_npy = np.reshape(data_train_npy, (-1, 100, 1))
    np.save('data_train.npy', data_train_npy)
    print("data_train")
    print(data_train_npy.shape)
    with open('data_train.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        print(data_train_npy,file=f)

def make_data_test():
    data_test_npy = np.array([np.concatenate(
        [np.expand_dims(data_generator.get_asymmetric_double_well_data(99),1),
        np.array([[0.0]] * 900)]) for i in range(500)])
    np.save('data_test.npy', data_test_npy)
    print("data_test")
    print(data_test_npy.shape)
    with open('data_test.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        print(data_test_npy,file=f)



def make_mask():
    mask_test_npy = np.array([[[1]]*100+[[0]]*900]*500)
    np.save('mask_test.npy', mask_test_npy)
    print("mask_test")
    print(mask_test_npy.shape)
    with open('mask_test.txt', 'w') as f:
        np.set_printoptions(threshold=np.inf)
        print(mask_test_npy,file=f)

make_data_train()
make_data_test()
make_mask()