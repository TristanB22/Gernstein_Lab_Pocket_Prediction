#############################################################
#															#
#		Protein-Ligand Pocket Prediction Program			#
#			Tristan Brigham & Alan Ianeselli				#
#															#
#############################################################

# this program is meant to initialize and train the UNet attention model for creating the ligand pocket
# predicting model 
# it will save the model to a new directory inside of the "Trained Models" directory once it is done training the model


import os
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from Unet_Attention_Model import PocketDataset, UNet, UNet_evaluating_custom_metric


# this should be defined at the start of the training run with platform logic
device = None

# the save directory that we are going to save the models to 
SAVE_DIRECTORY = "Trained_Models"

# the path to the file that contains all of the molecules that we are going to use to train themodel
TRAINING_FILEPATHS_FILE = "training_molecules_filepaths.txt"

# the learning rate for the model that we are using
LEARNING_RATE = 0.001

# the number of epochs that we are training for 
NUM_EPOCHS = 1

# the batch size the we are working with
BATCH_SIZE = 32

# whether we should shuffle while training
SHOULD_SHUFFLE = True

# the number of workers for the dataloader
# get the number of cpus to get the number of workers
num_cpus = os.cpu_count()
NUM_WORKERS = num_cpus

# adjust so we don't have too many
if NUM_WORKERS > 6:
	NUM_WORKERS = 6


# this function goes through the directory that we are saving the models to
# and looks for the first filename that is available that we can save the models to 
def find_available_filename(directory=SAVE_DIRECTORY, base_filename="UNet_"):

	# make sure that the directory exists
	os.makedirs(SAVE_DIRECTORY, exist_ok=True)

	# the number of the trained model
	number = 0

	# iterate and search for a new available name
	while True:

		# set the file name
		filename = f"{base_filename}{number}.pth"

		# create the full path
		full_path = os.path.join(directory, filename)

		# return the full path to save to
		if not os.path.exists(full_path):
			return full_path

		# if not
		number += 1

# this function actually trains the UNet model that we have defined with standard pytorch usage
def train_model(model, dataloader, epochs, optimizer, loss_fn, metric_fn, device="cpu"):

	# set the model to training mode
	model.train()

	# train for the specified number of epochs
	for epoch in range(epochs):
		
		# keep track of the running loss
		running_loss = 0.0
		running_metric = 0.0
		
		# process each of the inputs from the dataloader
		for inputs, targets in tqdm(dataloader, desc="Processing Batch"):

			# if the device that we are training on is mps, then we have to lower the precision of the float pointers
			# if device == "mps":
			# 	inputs = inputs.as_type(np.float32)
			# 	targets = targets.as_type(np.float32)
			
			# move everything to the device
			inputs = inputs.to(device, dtype=torch.float32)
			targets = targets.to(device, dtype=torch.float32)
			
			# reset optimizer
			optimizer.zero_grad()
			
			# forward pass
			outputs = model(inputs)

			# print(f"outputs shape: {outputs.shape}")
			# print(f"targets shape: {targets.shape}")
			
			# compute the loss
			loss = loss_fn(outputs, targets)
			
			# update everything
			loss.backward()
			optimizer.step()
			
			running_loss += loss.item()

			# evaluate the custom metric as well
			metric_value = metric_fn(targets, outputs)
			running_metric += metric_value.item()

		epoch_loss = running_loss / len(dataloader)
		epoch_metric = running_metric / len(dataloader)
		
		print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Metric: {epoch_metric:.4f}")


# this is the main function that we are going to call to get the program running off of the bat
def main():
	
	# check the type of device that we are running on to get the right usage
    # set up constants
	if torch.cuda.is_available():
		device = torch.device("cuda")
	elif torch.backends.mps.is_available():
		device = torch.device("mps")
		# device = torch.device("cpu")
	else:
		device = torch.device("cpu")

	print(f"Training model on {device}")

	# get the file paths from the filepaths text file
	filepaths = [line.strip() for line in open(TRAINING_FILEPATHS_FILE).readlines()]

	# initialize a dataset with the file paths
	dataset = PocketDataset(filepaths)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHOULD_SHUFFLE, num_workers=NUM_WORKERS)

	# initialize the unet using attention for training
	model = UNet(use_attn_2=True).to(device)
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
	loss_fn = nn.BCELoss()
	metric_fn = UNet_evaluating_custom_metric

	# train the model
	train_model(model, dataloader, epochs=NUM_EPOCHS, optimizer=optimizer, loss_fn=loss_fn, metric_fn=metric_fn, device=device)

	# get an available save path
	save_path = find_available_filename()

	print(f"Saving model to {save_path}")

	# save model
	torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
	main()