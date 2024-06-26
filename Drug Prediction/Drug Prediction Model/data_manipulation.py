# this contains all of the helper functions that are used in the main script
# we house all of the helper functions in a class called Utilities to keep the code organized

import os
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

# molecule processing packages
from rdkit import Chem
from rdkit.Chem import AllChem
from mdtraj import load

# creating the dataset
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# loading in environment variables
from dotenv import load_dotenv


# load the environment variables
load_dotenv()

    
	
class ProteinDoping(Dataset):

	'''
	This class is used to create the dataset that we are going to use to train the model. It is a subclass of the PyTorch
	Dataset class. The 
	'''

	def __init__(self, filepaths):
		self.filepaths = filepaths

	def __len__(self):
		return len(self.filepaths)

	def __getitem__(self, idx):
		path = self.filepaths[idx].strip()
		features, target = numpy_open_files_select_molecule_path(path)
		return features, target



class DrugDataLoader:

	'''
	This class effectively acts as a wrapper for the dataset defined above in order to facilitate the training and memory management
	of the model. It is used to instantiate the dataset that is going to be used to train the model and shuffle the dataset when necessary.
	'''

	def __init__(self, filename, mol_paths: list[str] = None, batch_size=32, load_percentage=0.3):

		'''
		This function is used to initialize the data loader object. It stores the information such as the paths to the files with the 
		molecules that we are interested in and the batch size that we are going to use to train the model.

		Parameters:
			mol_paths (list[str]): the list of paths to the molecules that we are going to use in this dataloader

		'''

		# check if the path overwrite is None for each of the two paths
		if mol_paths is None:

			# just load in all of the molecules (this could blow out the memory)
			data_path = os.getenv("TRAINING_DATA_PATH")
			
			with open(data_path, "r") as f:
				self.mol_paths = f.readlines()

		else:
			self.mol_paths = mol_paths


		# store the load_percentage and the batch size
		self.load_percentage = load_percentage
		self.batch_size = batch_size

		# instance variables with the suffixes for the features and pocket labels
		self.features_suffix = "_total_features.pt"
		self.target_suffix = "_total_target.pt"

		# finally, load in the data
		self.load_new_data()



	# function for iterating over the data
	def __iter__(self):
		return self.generator()

	# function for generating the data
	def generator(self):

		# open the file containing the paths to the data files
		with open(self.filename, 'r') as file:
			reader = csv.reader(file)
			batch = []
			for row in reader:
				batch.append(row)
				if len(batch) == self.batch_size:
					yield batch
					batch = []
			if batch:
				yield batch

	

	def load_new_data(self) -> None:
		
		''' 
		This function is used to reset the data instances that we have stored in the dataloader to change the data that the model
		that is training or being tested is exposed to. 
		
		Parameters:
			None

		Returns:
			None
		'''

		# figure out how many instances we are going to load in 
		num_train_instances = int(len(self.mol_paths) * self.load_percentage)

		# check that the number of instances is not zero
		if num_train_instances == 0:
			raise ValueError("The number of training instances is zero. Please check the load percentage and the number of instances in the data file.")
		
		# load in the training data
		self.train_data = []	

		# get the random indices for the data
		random_indices = np.random.choice(len(self.mol_paths), num_train_instances, replace=False)
		selected_train_molecule_paths = [self.mol_paths[i] for i in random_indices]

		# create the iterator depending on the verbosity
		if os.getenv("VERBOSITY_LEVEL") >= 1:
			train_data_iterator = tqdm(selected_train_molecule_paths, desc="Loading training data...")
		else:
			train_data_iterator = selected_train_molecule_paths

		# iterate through the training data and load in the data
		for t_path in train_data_iterator:
			self.train_data.append(self.__read_drug_data_file__(t_path))

		

	@staticmethod
	def __read_drug_data_file__(data_path: str) -> pd.DataFrame:
		
		'''
		This function is used to load the drug data from the data file that is passed in. It returns the attributes 
		of a single drug from the path that we give it and is a utility function for the class. 
		
		Parameters:
			data_path (str): the path to the data file that we want to load in
			
		Returns:
			tensor_set: a set of tensors that we loaded in from the data file
		'''
		
		# load in the data
		data = pd.read_csv(data_path)
		
		return data



# for testing the program
if __name__ == "__main__":
	
	# get the verbosity level
	verbosity_level = os.getenv("VERBOSITY_LEVEL")
	
	if verbosity_level >= 1:
		print("Starting utilities test script...")
	
	raise NotImplementedError("The utilities testing script has not been implemented yet.")
	
	# print the data
	print("The data has been loaded successfully!")
	
	# print the end of the program
	if verbosity_level >= 1:
		print("Ending utilities testing script...")