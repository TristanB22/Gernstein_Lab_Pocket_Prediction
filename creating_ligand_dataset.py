#############################################################
#															#
#		Protein-Ligand Pocket Prediction Program			#
#			Tristan Brigham & Alan Ianselli					#
#															#
#############################################################

# import all of the packages that we need for the program
from repo_imports import *



# this will be commented out and moved to the repo_imports file once we are done programming
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from rdkit import Chem
from rdkit.Chem import AllChem
from mdtraj import load

# for optimizing the search space and time of the ligand-protein distances
from scipy.spatial import cKDTree


# this class is used to pull all of the data from the directories that have the 
# information we need including directories with the pdb data
class MolecularData:
    
	# initializing the data loading object with the paths to the ligand pdb files and config
	# config_path: this is the path to the config file for the individual molecule that we are getting the structure and atoms for
	# ligand_path: the path to the file that contains the ligand information for the molecule above
	# pqr_path: path to the pqr data for the molecule (not needed most of the time since already integrated)
	def __init__(self, config_path, ligand_path, pqr_path=None):
		
		self.config_path = config_path
		self.ligand_path = ligand_path
		self.pqr_path = pqr_path
		
		# loading in the data
		self.config = load(config_path)
		self.ligand = load(ligand_path.replace(".mol2", ".pdb"))

		# load the pqr path if that is something that we need
		if pqr_path:
			self.xyz_pqr, self.charge, self.radius = self.open_pqr(pqr_path)


	# getting all of the atoms that are within some cutoff distance of the pocket
	# which we are going to classify as the atoms that create the pocket
	def get_pocket_indices(self, cutoff=0.45):
		
		# get all distances
		# to make the search for nearby atoms faster
		distances = cdist(self.config['xyz'][0], self.ligand['xyz'][0])

		# get the closest points to one another
		close_points = np.any(distances <= cutoff, axis=1)

		# find the pocket atoms for the ligand
		pocket_indices = set(tuple(coord) for coord, close in zip(self.config['xyz'][0], close_points) if close)
		return pocket_indices


	# gets the charge, radius, and coordinates for the molecule from the file that we specify
	# path_to_file: the path to the pqr file for the molecule that we consider
	def open_pqr(self, path_to_file):
		
		# open the file and read the lines
		with open(path_to_file) as file:
			lines = file.readlines()

		# init return arrays
		x_pqr, y_pqr, z_pqr, charge, radius = [], [], [], [], []

		# iterate through each of the lines to get the data
		for line in lines:

			# make sure that we are getting only the atom data
			if line.startswith("ATOM"):

				# add the data from each of the lines
				x_pqr.append(float(line[30:38]))
				y_pqr.append(float(line[38:46]))
				z_pqr.append(float(line[46:54]))
				charge.append(float(line.split()[-2]))
				radius.append(float(line.split()[-1]))

		# return the data in the right formats
		xyz_pqr = np.column_stack((x_pqr, y_pqr, z_pqr))
		return xyz_pqr, charge, radius





	# getting all of the atoms that are within some cutoff distance of the pocket
	# which we are going to classify as the atoms that create the pocket
	# this is the old method which we are going to use to benchmark
	def old_get_pocket_indices(self, cutoff=0.45):
		
		# check the distance is within the range
		pocket_indices = []
		
		# iterare through each of the coordinates in the ligand and the protein
		# and check whether the distance is minimized
		for protein_coords in self.config.xyz[0]:
			for ligand_coords in self.ligand.xyz[0]:
				if np.linalg.norm(protein_coords - ligand_coords) <= cutoff:
					pocket_indices.append(list(protein_coords))
		return list(set(map(tuple, pocket_indices)))  





# this helps us to see the tensors that we are working with
class TensorVisualizer:
    
	# initialize the number of bins that we are going to put the tensors into
	def __init__(self, nbins):
		self.nbins = nbins

	# create an individual tensor from the coordinates for the tensor
	# and the bins that we want to put them into
	def create_tensor(self, coordinates, axis_bins):
		
		# initialize the bins
		tensor = np.zeros((self.nbins, self.nbins, self.nbins))
		
		# iterate through the atoms and add them
		for coord in coordinates:
			x, y, z = coord
			
			# create a sub array
			bin_indices = [
				np.digitize([x], axis_bins[0])[0] - 1,
				np.digitize([y], axis_bins[1])[0] - 1,
				np.digitize([z], axis_bins[2])[0] - 1
			]
			
			# add the tuple to the total bins tensor
			tensor[tuple(bin_indices)] += 1
			
		return tensor

	# showing the tensor
	def plot_tensor(self, tensor):
		colors = np.zeros(tensor.shape + (3,))
		norm_tensor = (tensor - np.min(tensor)) / np.ptp(tensor)
		
		# red channel for the tensor
		colors[..., 0] = norm_tensor  
		
		# green channel for the tensor
		colors[..., 1] = 1 - norm_tensor  
		
		# blue channel of the tensor
		colors[..., 2] = 1 / (1 + (norm_tensor / (1 - norm_tensor))**(-2))  
		
		# make the figure and plot it
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.voxels(tensor, facecolors=colors, alpha=0.8)

		# show the tensor
		plt.show()


# define a main function for creating all of the tensors
def main():

	# get all of the refined molecules
	folders = os.listdir("refined-set/")

	# create the new dataset
	for folder in folders:

		# get the config path and the ligand path for the molecule
		config_path = f"refined-set/{folder}/{folder}_protein_cleaned.pdb"
		ligand_path = f"refined-set/{folder}/{folder}_ligand.mol2"

		# pull the molecular data
		molecular_data = MolecularData(config_path, ligand_path)

		# get the pocket indicies with the distance calculation
		pocket_indices = molecular_data.get_pocket_indices()

		# check if there are any pocket indices before proceeding
		if pocket_indices:
			
			# get the axis bins and put them together
			axis_bins = [np.linspace(min(coord), max(coord), 32 + 1) for coord in zip(*pocket_indices)]

			# visualize the bins for the output
			visualizer = TensorVisualizer(nbins=32)
			tensor = visualizer.create_tensor(pocket_indices, axis_bins)
			visualizer.plot_tensor(tensor)
		
		else:
			print(f"No pocket indices found for {folder}. Continuing to next folder.")



if __name__ == "__main__":
    
	main()