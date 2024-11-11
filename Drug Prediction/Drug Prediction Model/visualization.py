# visualization.py

import torch
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from graph_generation import MolGraphGeneration  

def visualize_molecule(mol, title='Molecule'):
    """
    Visualize an RDKit molecule.

	:param mol [rdkit.Chem.rdchem.Mol]: RDKit molecule to visualize
	:param title [str]: Title of the plot

	:ret [None]
    """

	# check that we actually input a molecule to show
    if mol is None:
        print("Invalid molecule, cannot visualize.")
        return
	
	# create the image of the molecule
	# and show it
    img = Draw.MolToImage(mol, size=(300, 300))
    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()


def visualize_dataset_sample(dataset, index=0):
	"""
	Visualize a molecule from the dataset.

	:param dataset [torch_geometric.data.Dataset]: Molecule dataset
	:param index [int]: Index of the molecule to visualize

	:ret [None]
	"""

	# get the molecule from the dataset	
	data = dataset[index]

	# visualize the molecule
	visualize_molecule(data, title=f'dataset molecule {index}')


def visualize_generated_molecule(generator, num_nodes):
	"""
	Generate and visualize a molecule using the generator.

	:param generator [torch.nn.Module]: Generator model
	:param num_nodes [int]: Number of nodes in the generated molecule

	:ret [None]
	"""

	# set the geneerator not to keep the gradient computationg graph
	generator.eval()

	with torch.no_grad():
		
		# collect the data for the molecule
		generated_data_list = generator(batch_size=1, num_nodes=num_nodes)
		
		# convert graphs to molecules
		mol_list = [MolGraphGeneration.graph_to_molecule(to_networkx(data, node_attrs=['x'], edge_attrs=['edge_attr'])) for data in generated_data_list]

		# filter out invalid molecules
		valid_mols = [mol for mol in mol_list if mol is not None]

		# check if we have any valid molecules
		if not valid_mols:
			print("No valid molecules generated.")
			return
		
		# sample the first valid one that we got and show it
		generated_mol = valid_mols[0]

		# show the molecule
		visualize_molecule(generated_mol, title='generated molecule')
