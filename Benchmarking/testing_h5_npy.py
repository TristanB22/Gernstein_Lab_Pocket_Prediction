# this program tests the loading and saving speeds of the numpy file 
# format against the h5 file format

import numpy as np
import h5py
import time
import os


# generate large random dataset for saving
def create_large_data(size):
    return np.random.rand(size, size)

# save the data that is input to npy format
def save_npy(data, filename):
    start_time = time.time()
    np.save(filename, data)
    return time.time() - start_time

# load the npy file formatted data
def load_npy(filename):
    start_time = time.time()
    data = np.load(filename)
    return time.time() - start_time, data

# save it to h5 format instead
def save_hdf5(data, filename):
    start_time = time.time()
    with h5py.File(filename, 'w') as f:
        f.create_dataset('dataset', data=data)
    return time.time() - start_time

# load from h5 with time stamp
def load_hdf5(filename):
    start_time = time.time()
    with h5py.File(filename, 'r') as f:
        data = f['dataset'][:]
    return time.time() - start_time, data

def main():
    
	# the size of the test dataset that we are using
    data_size = 10000  
    data = create_large_data(data_size)

    # save it to numpy with time report
    npy_file = 'test_data.npy'
    npy_save_time = save_npy(data, npy_file)
    npy_load_time, _ = load_npy(npy_file)

    # save to h5 with time result
    hdf5_file = 'test_data.h5'
    hdf5_save_time = save_hdf5(data, hdf5_file)
    hdf5_load_time, _ = load_hdf5(hdf5_file)

    # print the results
    print(f"NPY Save Time: {npy_save_time:.2f} seconds")
    print(f"NPY Load Time: {npy_load_time:.2f} seconds")
    print(f"HDF5 Save Time: {hdf5_save_time:.2f} seconds")
    print(f"HDF5 Load Time: {hdf5_load_time:.2f} seconds")

    # remove the testing files
    os.remove(npy_file)
    os.remove(hdf5_file)

if __name__ == '__main__':
    main()
