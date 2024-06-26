# this code is used to perform the initial training of the optimal drug prediction model

# import the necessary libraries
import os
import torch
import numpy as np
import pandas as pd

from dotenv import load_dotenv

# load the environment variables
load_dotenv()




if __name__ == "__main__":

	# get the verbosity level
	verbosity_level = os.getenv("VERBOSITY_LEVEL")
	
	if verbosity_level >= 1:
		print("Starting training script...")

	# load the data
	train_data_loader, test_data_loader = load_drug_prediction_data()