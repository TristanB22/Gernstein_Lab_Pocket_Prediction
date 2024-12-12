# data/utils.py


''' 
The purpose of this file is to define the utilities that we can use
to encode the molecular graph into a graph object that can be used
by the GNN model. The MolEncoder class is used to encode the atom and
bond features into a tensor. The MolGraphGeneration class is used to
generate the graph object from the molecular graph. The backbone atoms
are the atoms that are in the allowed atom types.
'''

import torch
import networkx as nx
from rdkit import Chem
from .constants import allowed_atom_types, max_valence_dict



class MolEncoder:

    '''
    MolEncoder class is used to encode the atom and bond features into a tensor
    that can be used downstream by the GNN model.
    '''

    # we defined this in another file
    atom_types = allowed_atom_types
    
    # get the bond types that are allowed in the molecule
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]

    # keeping track of the max possible degrees in the molecule
    max_possible_degree = 6
    min_formal_charge = -1
    max_formal_charge = 1

    @staticmethod
    def atom_features(atom):

        '''
        Function to encode the features of the atom into a tensor

        :param atom [rdkit.Chem.rdchem.Atom]: The atom that we are processing
        :ret [torch.tensor]: The features of the atom
        '''

        # get the atom type
        atom_type = atom.GetSymbol()

        # convert the atom type into a one hot tensor
        atom_type_one_hot = [1 if atom_type == t else 0 for t in MolEncoder.atom_types]

        # normalize the degree of the atom and the charges
        normalized_degree = atom.GetDegree() / MolEncoder.max_possible_degree
        formal_charge_normalized = (atom.GetFormalCharge() - MolEncoder.min_formal_charge) / (MolEncoder.max_formal_charge - MolEncoder.min_formal_charge)
        normalized_formal_charge = torch.tanh(torch.tensor(formal_charge_normalized, dtype=torch.float))


        # get the hybridization of the atom and turn it ton onehot tensor
        hybridization = atom.GetHybridization()
        hybridization_one_hot = [
            1 if hybridization == Chem.rdchem.HybridizationType.SP else 0,
            1 if hybridization == Chem.rdchem.HybridizationType.SP2 else 0,
            1 if hybridization == Chem.rdchem.HybridizationType.SP3 else 0
        ]

        # check if the atom is aromatic and normalize the valence
        # put this information into a tensor
        is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
        valence = max_valence_dict.get(atom_type, 4)
        normalized_valence = valence / 5.0
        
        # put all the features into a tensor
        features = [
            *atom_type_one_hot,
            normalized_degree,
            normalized_formal_charge,
            *hybridization_one_hot,
            is_aromatic,
            normalized_valence
        ]
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def bond_features(bond):

        '''
        Compute the features for the bonds in the molecule 
        that we are processing at a given time

        :param bond [rdkit.Chem.rdchem.Bond]: The bond that we are processing
        :ret [torch.tensor]: The features of the bond
        '''

        # get the bond type and turn it into a one hot tensor
        bt = bond.GetBondType()
        bond_type_one_hot = [
            1 if bt == Chem.rdchem.BondType.SINGLE else 0,
            1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
            1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
            1 if bt == Chem.rdchem.BondType.AROMATIC else 0
        ]

        # check if the bond is conjugated and if it is in a ring
        is_conjugated = 1.0 if bond.GetIsConjugated() else 0.0
        is_in_ring = 1.0 if bond.IsInRing() else 0.0

        # put all the features into a tensor
        bond_features = [
            *bond_type_one_hot,
            is_conjugated,
            is_in_ring
        ]

        return torch.tensor(bond_features, dtype=torch.float)

    @staticmethod
    def denormalize_formal_charge(normalized_formal_charge):
        
        '''
        Denormalize the formal charge of the atom. This can be implemented
        in the case that we want to get the formal charge of the atom in the
        molecule later on with new normalization strategies and data. 

        :param normalized_formal_charge [float]: The normalized formal charge of the atom
        :ret [int]: The denormalized formal charge of the atom
        '''

        return normalized_formal_charge

    @staticmethod
    def denormalize_degree(normalized_degree):

        '''
        Denormalize the degree of the atom using the max possible degree
        that we have defined in the class. 

        :param normalized_degree [float]: The normalized degree of the atom
        :ret [int]: The denormalized degree of the atom
        '''

        # denormalize the degree 
        degree = int(normalized_degree * MolEncoder.max_possible_degree)
        return degree


class MolGraphGeneration:

    '''
    MolGraphGeneration class is used to generate the graph object from the molecular graph
    '''

    @staticmethod
    def get_backbone_graph(mol):

        '''
        Get the backbone of the molecule and generate the graph object
        from the backbone atoms.

        :param mol [rdkit.Chem.rdchem.Mol]: The molecule that we are processing
        :ret [nx.Graph]: The graph object of the backbone atoms
        '''

        # create a graph object for constructing the backbone in
        G = nx.Graph()

        # get the backbone atoms
        backbone_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in allowed_atom_types]

        # if there are no backbone atoms, return None
        if len(backbone_atoms) == 0:
            return None
        
        # add the nodes and edges to the graph
        for atom_idx in backbone_atoms:

            # get the atom and add it to the graph using the index 
            # of the atom
            atom = mol.GetAtomWithIdx(atom_idx)
            G.add_node(atom_idx, features=MolEncoder.atom_features(atom))

        # add the edges to the graph
        for bond in mol.GetBonds():
            
            # get the begin and end atoms of the bond
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()

            # if the atoms are in the backbone, add the edge to the graph
            if u in backbone_atoms and v in backbone_atoms:
                G.add_edge(u, v, features=MolEncoder.bond_features(bond))

        # return the graph object if there are nodes in the graph
        if G.number_of_nodes() > 0:
            return G
        else:
            return None
