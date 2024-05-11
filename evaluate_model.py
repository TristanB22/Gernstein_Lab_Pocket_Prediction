
#############################################################
#															#
#		Protein-Ligand Pocket Prediction Program			#
#			Tristan Brigham & Alan Ianeselli				#
#															#
#############################################################

# this file is used to evaluate a trained and saved model 
import os
import torch
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# import the model structure and definition
from Unet_Attention_Model import UNet, open_files, obtain_coordinates, visualize_protein


# the model that we are going to load and evaluate
MODEL_PATH = "./Trained_Models/UNet_0.pth"

# path to all of the molecules training data
MOLECULES_PATH = "refined-set"

# the name of the protein that we are evaluating against
PDB = "3qqs"


# this loads the model from the path that is given to the function
# model_path: the path to the modle that we want to load
# model_class: the class and structure of the model that we want to evaluate
# device: the device that we should put the model on 
def load_model(model_path, model_class, device='cpu'):
    
	# create the model
	loaded_model = model_class()
		
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

# fucntion for actually doing the evaluation of the model
def evaluate_model(model, loader, device='cpu'):
    
	# set the model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        
		# iterate through the evaluation dataset
        for data, target in loader:
            
			# move the data to the right device
            data, target = data.to(device), target.to(device)
            output = model(data)
            
    return output

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
     
	# load in the molecule that we are going to use to evaluate the model
	features, target = open_files(os.path.join(MOLECULES_PATH, PDB))
     
	# get the features and target tensors
	dataset = [(features, target)]
     
	# define a dataloader for the dataset that we are using
	loader = DataLoader(dataset, batch_size=1, shuffle=False)
     
	# get the predicted pocket from the model
	predicted_tensor = evaluate_model(model, loader, device=device)
    
	# getting the visualization coordinates
	xyz_protein, predicted_values, xyz_pocket_target = obtain_coordinates(PDB, predicted_tensor.squeeze())

	# Visualize the result
	visualize_protein(xyz_protein, predicted_values, xyz_pocket_target, PDB)


# run the evaluation of the model
if __name__ == "__main__":
    main()