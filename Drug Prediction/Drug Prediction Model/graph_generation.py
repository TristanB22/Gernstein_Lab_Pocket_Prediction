import os
import torch
import argparse
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx


class MolEncoder:
    """
    Encodes molecules with features necessary for graph generation, such as atom type, charge, and bond types.
    """
    
    atom_types = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl',
        'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 
        'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    ]

    @staticmethod
    def atom_features(atom: Chem.Atom) -> torch.Tensor:
        """
		Create a feature vector for an atom
		"""

		# get the type of the atom with its symbol
        atom_type = atom.GetSymbol()

		# get the index of the atom type in the atom types list
        atom_type_idx = MolEncoder.atom_types.index(atom_type) if atom_type in MolEncoder.atom_types else -1

		# encode the features of the atom
		# which includes the atom type index, the atomic number, the degree of the atom, the formal charge of the atom,
		# the hybridization of the atom, whether the atom is aromatic, and the chiral tag of the atom
        features = [
            atom_type_idx, atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            atom.GetHybridization().real, atom.GetIsAromatic(), atom.GetChiralTag()
        ]

		# return the atom features in tensor format
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def bond_features(bond: Chem.Bond) -> torch.Tensor:
        """
		Create a feature vector for a bond.

		:param bond [Chem.Bond]: RDKit bond object that this function is analyzing

		:ret [torch.Tensor]: Feature vector for the bond 
		"""

		# getting the type of the bond
        bt = bond.GetBondType()

		# return a tensor with the features of the bond
		# to keep everything in the computation scheme of torch
        return torch.tensor([
            bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(), bond.IsInRing()
        ], dtype=torch.float)



class MolGraphGeneration:

	'''
	Class for generating the graphs that we are going to use for representing the molecules in the dataset
	'''

	@staticmethod
	def get_backbone_graph(mol: Chem.Mol) -> nx.Graph:
		"""
		Extract the backbone graph from a molecule, including only non-hydrogen atoms and relevant bonds.
		
		:param mol [Chem.Mol]: RDKit molecule object to extract the backbone graph from

		:ret [nx.Graph]: NetworkX graph object representing the backbone of the molecule
		"""

		# create a graph object
		G = nx.Graph()

		# get the indices of the atoms that are not hydrogen atoms
		# which are going to form the backbone of the molecule that we create
		backbone_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() != 'H'}
		
		# add the atoms to the graph
		for atom_idx in backbone_atoms:

			# get the atom object
			atom = mol.GetAtomWithIdx(atom_idx)

			# add to the graph
			G.add_node(atom_idx, features=MolEncoder.atom_features(atom))
		
		# get all of the bonds that we could consider in the molecule
		for bond in mol.GetBonds():

			# indicies of the atoms that are connected by the bond
			u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

			# add the edge if both atoms are in the backbone
			if u in backbone_atoms and v in backbone_atoms:
				G.add_edge(u, v, features=MolEncoder.bond_features(bond))
		
		# return the graph
		return G if G.nodes else None


	@staticmethod
	def load_all_sdf_files(root_dir: str) -> list:
		"""
		Load and process all .sdf files in the specified directory, extracting backbone graphs. We are only using
		the sdf files to define the ligandso over the course of this project so we are only going to be considering
		those backbone atoms for the molecule. 
		
		:param root_dir [str]: Path to the directory containing .sdf files
		
		:ret [list[NetworkX]]: List of NetworkX graph objects representing the backbone of each molecule
		"""

		# the list of graphs that we get
		graphs = []

		# go through each of the subdirectories of the root directory
		# to look for the sdf files that we are going to use for the projec
		for subdir, _, files in os.walk(root_dir):

			# go through each of the files in the subdirectory
			for file in files:

				# check if the file is an sdf file
				if file.endswith('.sdf'):

					# get the path to the file
					# and load it in
					file_path = os.path.join(subdir, file)
					supplier = Chem.SDMolSupplier(file_path, sanitize=True, removeHs=False)
					for mol in supplier:
						
						# check if the molecule that we got is valid or not
						if mol:
							backbone_graph = MolGraphGeneration.get_backbone_graph(mol)
						
							# if we got a backbone for the molecule then save that
							if backbone_graph:
								graphs.append(backbone_graph)
		return graphs


	@staticmethod
	def molecule_to_graph(molecule: Chem.Mol) -> nx.Graph:
		"""
		Convert an RDKit molecule object to a NetworkX graph.
		
		:param molecule [Chem.Mol]: RDKit molecule object to convert to a graph
		
		:ret [nx.Graph]: NetworkX graph object representing the molecule
		"""

		# create a graph object
		G = nx.Graph()

		# add the atoms to the graph
		for atom in molecule.GetAtoms():
			G.add_node(atom.GetIdx(), features=MolEncoder.atom_features(atom))
		
		# add the bonds to the graph
		for bond in molecule.GetBonds():
			G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), features=MolEncoder.bond_features(bond))
		
		return G


	@staticmethod
	def graph_to_molecule(graph: nx.Graph) -> Chem.Mol:
		"""
		Convert a NetworkX graph back to an RDKit molecule object and add hydrogens.
		
		:param graph [nx.Graph]: NetworkX graph object to convert to an RDKit molecule
		
		:ret [Chem.Mol]: RDKit molecule object representing the graph
		"""

		# create a writable molecule object
		mol = Chem.RWMol()

		# add the atoms into the chem obj
		for node_idx, attr in graph.nodes(data=True):
			
			# get the features of the atom
			features = attr.get('features', torch.tensor([0], dtype=torch.float))

			# scale the atom type index to the valid range
			num_valid_atom_types = len(MolEncoder.atom_types) - 1  

			# get the index of the atom type
			atom_type_idx = int(round(features[0].item() * (num_valid_atom_types - 1)))

			# make sure that the index is valid
			atom_type_idx = max(0, min(atom_type_idx, num_valid_atom_types - 1))

			# checking what the atom type is
			atom_type = MolEncoder.atom_types[atom_type_idx]
			atom = Chem.Atom(atom_type)
			
			# add the atom to the molecule
			mol.AddAtom(atom)
		
		
		# add bonds to the grahph
		for u, v, attr in graph.edges(data=True):
		
			# get the features of the bond
			bond_features = attr.get('features', torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float))
			bond_type_idx = int(round(bond_features[0].item() * 3))  

			# get the bond type
			bond_type = {
				0: Chem.rdchem.BondType.SINGLE,
				1: Chem.rdchem.BondType.DOUBLE,
				2: Chem.rdchem.BondType.TRIPLE,
				3: Chem.rdchem.BondType.AROMATIC
			}.get(bond_type_idx, Chem.rdchem.BondType.SINGLE)

			# check whether the bond already exists or not
			if mol.GetBondBetweenAtoms(int(u), int(v)) is not None:
				continue
			
			# add the bond if it does not already exist in the molecule
			mol.AddBond(int(u), int(v), bond_type)

		try:

			# convert the molecule to a mol object
			mol = mol.GetMol()
			Chem.SanitizeMol(mol)
		
			# add hydrogens to the molecule
			return Chem.AddHs(mol)
		
		except Exception as e:
			print(f"Error in graph_to_molecule: {e}")
			return None



def main():

	# create the argument parser
    parser = argparse.ArgumentParser(description="Molecule processing script")
    parser.add_argument('--visualize', type=str, help="Path to the SDF file to visualize")
    parser.add_argument('--fragment', type=str, help="Path to the SDF file or directory to fragment")
    args = parser.parse_args()

	# visualize the molecule if that is something
	# that we want to do according to the arguments
    if args.visualize:
        sdf_file_path = args.visualize
        visualize_molecule_from_sdf(sdf_file_path, mol_index=0, title="Example Molecule")

	# fragment the molecule if that is something that we want to do
    elif args.fragment:
        directory_path = args.fragment
        all_graphs = MolGraphGeneration.load_all_sdf_files(directory_path)
        print(f"Total graphs created: {len(all_graphs)}")

    else:
        print("Please specify --visualize or --fragment as arguments.")



def visualize_molecule_from_sdf(sdf_path: str, mol_index=0, title="Molecule from SDF"):
	"""
	Visualize a molecule from an SDF file.

	:param sdf_path [str]: Path to the SDF file
	:param mol_index [int]: Index of the molecule to visualize
	:param title [str]: Title of the plot

	:ret [None]
	"""

	# load the molecule from the sdf file
	supplier = Chem.SDMolSupplier(sdf_path)

	try:

		# get the right molecule according to the index
		mol = supplier[mol_index]
		
		# visualize the molecule
		if mol:

			# create the image of the molecule
			img = Draw.MolToImage(mol, size=(300, 300))
			plt.imshow(img)
			plt.axis('off')
			plt.title(title)
			plt.show()
		else:
			print("Invalid molecule at specified index.")
	except Exception as e:
		print(f"Error in visualization: {e}")


if __name__ == "__main__":
    main()
