
#############################################################
#															#
#		Protein-Ligand Pocket Prediction Program			#
#			Tristan Brigham & Alan Ianeselli				#
#															#
#############################################################

# this file is used to evaluate a trained and saved model 
import os
import shap
import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# import the model structure and definition
from Unet_Attention_Model import UNet, torch_open_files_select_molecule_path, obtain_coordinates, visualize_protein


# are we loading in a model that used attention or not?
USE_ATTN = True

# whether we should evaluate with a bunch of molecules or only one
EVALUATE_ALL = True

# the path for evaluating the model
EVALUATING_PATH = "training_molecules_filepaths.txt"

# the model that we are going to load and evaluate
MODEL_PATH = "./Trained_Models/UNet_1.pth"

# path to all of the molecules training data
# MOLECULES_PATH = "refined-set"
MOLECULES_PATH = "'/home/tjb73/palmer_scratch/refined-set"

# the name of the protein that we are evaluating against
PDB = "3qqs"


# this loads the model from the path that is given to the function
# model_path: the path to the modle that we want to load
# device: the device that we should put the model on 
def load_model(model_path, device='cpu'):
    
	# create the model
	loaded_model = UNet(use_attn_2=USE_ATTN)
		
	# load the model in 
	loaded_model.load_state_dict(torch.load(model_path, map_location=device))
	
	# move the model
	loaded_model = loaded_model.to(device)

	return loaded_model

# loading the training history of the model in from the pkl file 
# if that is something that we have saved
# history_path: the path to the pickled history file
def load_history(history_path):
    
	# open the path to the history
    with open(history_path, 'rb') as f:
        history_data = pkl.load(f)
        
	# return it
    return history_data




# function for evaluating the model and computing Shapley values for each input feature
# to get the feature importance
# model: the model that we are going to evaluate
# loader: the dataloader that we are going to use to evaluate the model
# device: the computing device ('cpu' or 'cuda')
def evaluate_model_with_shap(model, loader, device='cpu'):
    
    # set the model to evaluation mode
    model.eval()

    # initialize the SHAP explainer using a subset of data (assuming data from loader can fit into memory)
    background_data = next(iter(loader))[0][:100].to(device)  # use the first 100 examples to estimate background distribution
    explainer = shap.DeepExplainer(model, background_data)

    # arrays we are returning
    predictions = []
    shap_values = []
    total_mse = 0
    num_samples = 0

    with torch.no_grad():
        
        # iterate through the evaluation dataset
        for data, target in loader:
            
            # move the data and target to the right device
            data, target = data.to(device), target.to(device)
            predicted_tensor = model(data)

            # calculate the mean squared error
            mse = torch.mean((predicted_tensor - target) ** 2)
            total_mse += mse.item()
            num_samples += 1

            # compute SHAP values for this batch
            shap_values_batch = explainer.shap_values(data)
            shap_values.append(shap_values_batch)

            # store predictions
            predictions.append(predicted_tensor)
            
    # normalize the mse by the number of samples
    normalized_mse = total_mse / num_samples

    # return all of the predictions, normalized mse, and SHAP values for all samples
    return predictions, normalized_mse, shap_values



# fucntion for actually doing the evaluation of the model
# model: the model that we are going to evaluate
# loader: the dataloader that we are going to use to evaluate the model
# visualize: whether we should visualize the output of the model (if this option is on, we only evaluate one molecule)
def evaluate_model(model, loader, visualize=False, device='cpu'):
    
	# set the model to evaluation mode
	model.eval()

	# arrays we are returning
	predictions = []
	total_mse = 0
	num_samples = 0

	with torch.no_grad():
		
		# iterate through the evaluation dataset
		for data, target in loader:
			
			# move the data to the right device
			data, target = data.to(device), target.to(device)
			predicted_tensor = model(data)

			# calculate the mean squared error
			mse = torch.mean((predicted_tensor - target) ** 2)
			total_mse += mse.item()
			num_samples += 1

			# if we should be visualizing the output
			if visualize:
				
				# getting the visualization coordinates
				xyz_protein, predicted_values, xyz_pocket_target = obtain_coordinates(PDB, predicted_tensor.squeeze())

				# visualize the result
				visualize_protein(xyz_protein, predicted_values, xyz_pocket_target, PDB)

				# return the single prediction
				return [predicted_tensor], total_mse
			
			# otherwise push it to the array we are returning
			predictions.append(predicted_tensor)
			
	# normalize the mse by the number of samples
	normalized_mse = total_mse / num_samples

	# return all of the predictions and the normalized mse
	return predictions, normalized_mse



# define the main evaluation function
def main():
     
	# gettign the device that we should evaluate on
	if torch.cuda.is_available():
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
	else:
		device = torch.device("cpu")

	print(f"Training model on {device}")
     

	# pull the model
	model = load_model(MODEL_PATH, device=device)
     
	# should we evaluate all of the molecules or just one
	if EVALUATE_ALL:

		# we are going to evaluate all of the molecules
		# get the file paths from the filepaths text file
		filepaths = [line.strip() for line in open(EVALUATING_PATH).readlines()]

		# initialize a dataset with the file paths
		dataset = [torch_open_files_select_molecule_path(path) for path in filepaths]

		# define a dataloader for the dataset that we are using
		loader = DataLoader(dataset, batch_size=1, shuffle=False)

	else:
		
		# load in the molecule that we are going to use to evaluate the model
		features, target = torch_open_files_select_molecule_path(os.path.join(MOLECULES_PATH, PDB))
		
		# get the features and target tensors
		dataset = [(features, target)]
		
		# define a dataloader for the dataset that we are using
		loader = DataLoader(dataset, batch_size=1, shuffle=False)
     
	# get the predicted pocket from the model
	predictions, normalized_mse, shap_values = evaluate_model_with_shap(model, loader, device=device)

	# print the total loss
	print(f"MSE Loss: {normalized_mse}")

	# now print the shap values for each of the input channels
	model_channels = ['N', 'bfactors', 'buriedness', 'charge', 'radius', 'hbdon', 'hbacc', 'sasa']
	
	# print the SHAP vals with the factor
	for m_channel, s_val in zip(model_channels, shap_values):

		# print the SHAP vals with the factor
		for m_channel, s_val in zip(model_channels, shap_values):
			
			# print it
			print(f"SHAP values for {m_channel}: {s_val}")


# run the evaluation of the model
if __name__ == "__main__":
    main()