import os
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import time

# this loads in a tensor from a numpy file and converts it to a pytorch tensor
# path: the path to the file that has the tensor that we are loading
def load_numpy(path):
    return torch.tensor(np.load(path).astype(np.float32 ))

def standard_scaler(non_zero_indices, tensor):

	# standardize only the non-zero elements
	mean = torch.mean(tensor[non_zero_indices])
	std = torch.std(tensor[non_zero_indices])

	# add a small constant for numerical stability
	tensor[non_zero_indices] = (tensor[non_zero_indices] - mean) / (std + 1e-8)  

	return tensor

def min_max_scaler(non_zero_indices, tensor):
	
	# max and min vals
	max_ = torch.max(tensor[non_zero_indices])
	min_ = torch.min(tensor[non_zero_indices])

	# scale only the non-zero elements
	# add a small constant to avoid division by zero
	tensor[non_zero_indices] = (tensor[non_zero_indices] - min_) / (max_ - min_ + 1e-8)  

	return tensor

def   mean_max_scaler(non_zero_indices, tensor):
    
	# max and min values
	max_ = torch.max(tensor[non_zero_indices])
	min_ = torch.min(tensor[non_zero_indices])
	mean_ = torch.mean(tensor[non_zero_indices])
     
	# scale and shift only the non-zero elements
	# add a small constant for stability
	tensor[non_zero_indices] = (tensor[non_zero_indices] - mean_) / (max_ - min_ + 1e-8)  
     
	return tensor

def max_scaler(non_zero_indices, tensor):
    
	# get max value
	max_ = torch.max(tensor[non_zero_indices])

	# normalize only non-zero elements to their maximum value
	# prevent division by zero
	tensor[non_zero_indices] = tensor[non_zero_indices] / (max_ + 1e-8)  

	return tensor

# the custom metric that we are using
def UNet_evaluating_custom_metric(y_true, y_pred, eta = 1e-11):
    
	# get the binary mask
    mask = y_true > eta
    
	# return the mean and abs of diff
    return torch.mean(torch.abs(y_true[mask] - y_pred[mask]))



# this function loads in all of the data that is associated with a molecule
# it is optimized for the old versions of the files that used numpy tensors
# given the path and returns the features + labels
# path: the path to the molecule that we are trying to load in
# parent_dir: the parent directory where the directory of molecules is
def numpy_open_files_select_molecule_path(path, parent_dir=False):
	
	# non-zero value to replace zeros with
	nzv = 0.0  

	# update path to be relative
	# path = os.path.join("..", path)

	# open the features of the observation
	# ensure it is a float tensor
	N = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/N_tensor.npy")).astype(np.float32))

	# indices of non-zero values (i.e., protein)
	non_zeros = torch.nonzero(N, as_tuple=True)  
	
	# indices of zero values (i.e., outside)
	zeros = torch.nonzero(N == 0, as_tuple=True) 

	# set protein to 1
	N[non_zeros] = 1.0  
	
	# set outside to nzv
	N[zeros] = nzv  

	# add an extra dimension for the later merging, to create the batch
	N = N.unsqueeze(-1)  

	bfactors = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/bfactors_tensor.npy")).astype(np.float32))
	# bfactors = bfactors #make sure it is a float tensor
	bfactors = mean_max_scaler(non_zeros, bfactors) #normalize between 0 and 1 (+ outliers)
	bfactors[zeros] = nzv #outside gets nzv
	bfactors = bfactors.unsqueeze(-1)

	#repeat for all the other features
	buriedness = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/buriedness_tensor.npy")).astype(np.float32))
	# buriedness = buriedness 
	buriedness = mean_max_scaler(non_zeros, buriedness)
	buriedness[zeros] = nzv
	buriedness = buriedness.unsqueeze(-1)

	charge = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/charge_tensor.npy")).astype(np.float32))
	# charge = charge
	charge = mean_max_scaler(non_zeros, charge)
	charge[zeros] = nzv
	charge = charge.unsqueeze(-1)

	radius = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/radius_tensor.npy")).astype(np.float32))
	# radius = radius 
	radius = mean_max_scaler(non_zeros, radius)
	radius[zeros] = nzv
	radius = radius.unsqueeze(-1)

	hbdon = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/hbdon_tensor.npy")).astype(np.float32))
	# hbdon = hbdon 
	hbdon = mean_max_scaler(non_zeros, hbdon)
	hbdon[zeros] = nzv
	hbdon = hbdon.unsqueeze(-1)

	hbacc = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/hbac_tensor.npy")).astype(np.float32))
	# hbacc = hbacc 
	hbacc = mean_max_scaler(non_zeros, hbacc)
	hbacc[zeros] = nzv
	hbacc = hbacc.unsqueeze(-1)

	sasa = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/sasa_tensor.npy")).astype(np.float32))
	# sasa = sasa 
	sasa = mean_max_scaler(non_zeros, sasa) #NOTE: could consider recalculating the non_zero indices just for sasa, as the inside of the protein contains zeros
	sasa[zeros]= nzv
	sasa = sasa.unsqueeze(-1)

	# get all of the tensors together now
	features = torch.cat([N, bfactors, buriedness, charge, radius, hbdon, hbacc, sasa], dim=-1)

	# target feature
	target=torch.from_numpy(np.load(os.path.join(path, "for_labview_pocket_/N_tensor_new_pocket.npy")))
	non_zeros_target = (torch.nonzero(target, as_tuple=True)) #calculate the non_zero values of the target (i.e. pocket)

	eta=0.00000000001
	target[non_zeros]+= eta  #to distinguish it from the zeros of the outside (for metrics calculation)
	
	# make it binary, the pocket becomes 1
	target[non_zeros_target] = 1.0 
      
	# everything that is outside in the observation becomes nvz, also to mask out the wrong hydrogens added to the pocket
	target[zeros] = nzv 

	# add the batch dimension to the outside
	target = target.unsqueeze(-1)

	return features, target




# this function loads in all of the data that is associated with a molecule
# it is optimized for the old versions of the files that used numpy tensors
# given the path and returns the features + labels
# path: the path to the molecule that we are trying to load in
# parent_dir: the parent directory where the directory of molecules is
def torch_open_files_select_molecule_path(path, parent_dir=False):
	
	features = torch.load(os.path.join(path, f"{path.replace('../refined-set/', '')}_total_features.pt"))
	target = torch.load(os.path.join(path, f"{path.replace('../refined-set/', '')}_total_target.pt"))

	return features, target


# converting the entire tensor to a single file
def convert_all_features(path, parent_dir=False):

	# get the features and the target
	features, target = numpy_open_files_select_molecule_path(path)

	# now save the features and target to separate files
	torch.save(features, os.path.join(path, f"{path.replace('../refined-set/', '')}_total_features.pt"))
	torch.save(target, os.path.join(path, f"{path.replace('../refined-set/', '')}_total_target.pt"))



testing_path = "../refined-set/1a1e"

# testing the load speeds of the two files
if __name__ == "__main__":

	# first, we need to ensure the pytorch versions of the files exist for a fair comparison
	convert_all_features(testing_path)

	# measure the time for the numpy-based loading function
	start_time_numpy = time.time()
	numpy_features, numpy_target = numpy_open_files_select_molecule_path(testing_path)
	end_time_numpy = time.time()

	# measure the time for the torch-based loading function
	start_time_torch = time.time()
	torch_features, torch_target = torch_open_files_select_molecule_path(testing_path)
	end_time_torch = time.time()

	# calculate the duration of each function
	duration_numpy = end_time_numpy - start_time_numpy
	duration_torch = end_time_torch - start_time_torch

	# print the results
	print(f"duration using numpy: {duration_numpy:.4f} seconds")
	print(f"duration using torch: {duration_torch:.4f} seconds")

	# calculate and print the speedup
	if duration_torch > 0:  # prevent division by zero
		speedup = duration_numpy / duration_torch
		print(f"speedup from numpy to torch: {speedup:.2f}x")
	else:
		print("no measurable duration for torch-based loading to calculate speedup.")
