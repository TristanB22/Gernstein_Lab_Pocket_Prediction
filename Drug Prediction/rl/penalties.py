# rl/penalties.py

'''
Computing the penalties that should be applied to a molecule 
based on its properties.
'''

import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import Descriptors
import networkx as nx
from data.constants import max_valence_dict
from config import (DISCONNECT_PENALTY, DISTRIBUTION_PENALTY, 
                    SAME_ATOM_PENALTY, DUPLICATE_PENALTY, EDGE_DENSITY_PENALTY, 
                    VALENCE_PENALTY_SCALING, SIZE_SCALER, EDGE_COUNT_PENALTY_SCALING,
                    DIST_DISTRIBUTION_PENALTY, INVALID_EDGES_SCALING)

def compute_disconnect_penalty(mol):
    
    '''
    Add a penalty that scales with the number of components in the molecule
    that are disconnected.

    :param mol [rdkit.Chem.Mol]: The molecule to compute the penalty for.
    :return [float]: The penalty to apply to the molecule.
    '''

    # check that the mol is legit
    if mol is None:
        return 0.0
    
    # get the number of components in the molecule
    num_components = Chem.GetNumAtoms(mol) - Chem.GetNumBonds(mol) + Chem.GetSSSR(mol) + 1
    
    # scale the penalty by the number of components
    if num_components > 1:
        return DISCONNECT_PENALTY * (num_components - 1)
    
    return 0.0

def compute_distribution_penalty(mol):
    
    '''
    Add a penalty that scales with the deviation of the molecular weight
    from the desired molecular weight.

    :param mol [rdkit.Chem.Mol]: The molecule to compute the penalty for.
    :return [float]: The penalty to apply to the molecule.
    '''

    # check that the mol is legit
    if mol is None:
        return 0.0
    
    # get the molecular weight of the molecule
    # and scale it
    mw = Descriptors.MolWt(mol)
    desired_mw = 300  
    deviation = abs(mw - desired_mw)
    return DIST_DISTRIBUTION_PENALTY * deviation


def compute_same_atom_penalty(mol):
    
    '''
    Add a penalty in the case that we find molecules that are entirely made of the same atom
    so that we penalize molecules that suffer from mean or mode collapse. 

    :param mol [rdkit.Chem.Mol]: The molecule to compute the penalty for.
    :return [float]: The penalty to apply to the molecule.
    '''

    # check that the mol is legit
    if mol is None:
        return 0.0
    
    # get the atom counts
    from collections import Counter

    # get the distribution of atoms in the molecule
    atom_counts = Counter([atom.GetSymbol() for atom in mol.GetAtoms()])
    total_atoms = mol.GetNumAtoms()
    
    # keep track of the penalty
    penalty = 0.0

    # make sure that none of the atoms overrepresents the atom types in a given molecule
    for atom, count in atom_counts.items():
        
        # get the percentage of the atom in the molecule
        percentage = (count / total_atoms) * 100

        # if the percentage is greater than 50, we apply a penalty
        # that scales with the amount over 50 percent tha this represents
        if percentage > 50:
            excess_percentage = percentage - 50
            penalty += SAME_ATOM_PENALTY * (2 ** (excess_percentage / 50))

    return penalty


def compute_duplicate_penalty(mol):
        
    '''
    Add a penalty to the model if we find tha this is a molecule
    that has already been generated. This is something that is left up to the
    user to implement down the line. 

    :param mol [rdkit.Chem.Mol]: The molecule to compute the penalty for.
    :return [float]: The penalty to apply to the molecule.
    '''

    # not implemented right now
    return 0.0

def compute_edge_density_penalty(mol):
    
    '''
    Computed the edge density penalty for a molecule. This is the difference between the
    actual edge density and the desired edge density.

    :param mol [rdkit.Chem.Mol]: The molecule to compute the penalty for.
    :return [float]: The penalty to apply to the molecule.
    '''
    
    # check that the mol that is input is 
    # a valid molecule
    if mol is None:
        return 0.0
    
    # get the number of atoms in the molecule
    # and check that we can even compute the edge density
    num_atoms = mol.GetNumAtoms()
    if num_atoms < 2:
        return 0.0
    
    # get the total number of possible edges
    total_possible_edges = num_atoms * (num_atoms - 1) / 2
    
    # get the actual number of edges
    actual_edges = mol.GetNumBonds()
    
    # we are targeting roughly a density of 3
    # and penalize larger deviations from that
    desired_edge_density = 3 / total_possible_edges if total_possible_edges > 0 else 0

    # compute the edge density and apply the penalty
    edge_density = actual_edges / total_possible_edges
    return EDGE_DENSITY_PENALTY * max(0.0, edge_density - desired_edge_density)


def compute_invalid_edge_penalty(mol):
    
    '''
    Compute the penalty for invalid edges in the molecule. This is done by checking
    the bond types in the molecule and seeing if they are valid or not.

    :param mol [rdkit.Chem.Mol]: The molecule to compute the penalty for.
    :return [float]: The penalty to apply to the molecule.
    '''

    from data.utils import MolEncoder

    # check that the mol that is input is
    if mol is None:
        return 0.0
    
    # get the invalid bonds in the molecule
    invalid_bonds = [bond for bond in mol.GetBonds() if bond.GetBondType() not in MolEncoder.bond_types]
    num_invalid_edges = len(invalid_bonds)
    
    # return the penalty
    return INVALID_EDGES_SCALING * num_invalid_edges


def count_invalid_valences(mol):
    
    '''
    Count the number of invalid valences in a molecule.

    :param mol [rdkit.Chem.Mol]: The molecule to compute the penalty for.
    :return [int]: The number of invalid valences in the molecule.
    '''

    # check that the mol that is input is
    if mol is None:
        return 0
    
    # keep track of the nnumber of invalid valence electrons that we have in the model
    invalid_count = 0

    # get the atom types in the molecule
    for atom in mol.GetAtoms():
        
        # get the atom type and the maximum valence
        atom_type = atom.GetSymbol()

        # check for atom types that are not 
        # in the max valence dictionary
        if atom_type not in max_valence_dict:
            raise ValueError(f"Unknown atom type '{atom_type}' in molecule.")
        
        # get the maximum valence for the atom
        max_val = max_valence_dict[atom_type]

        # get the valence of the atom
        val = atom.GetExplicitValence() + atom.GetImplicitValence()

        # check if the valence is greater than the maximum valence
        if val > max_val:
            
            # scale the penalty
            invalid_count += (val - max_val)
        
    return invalid_count


def compute_edge_count_penalty(mol):
    
    '''
    Compute the penalty for the number of edges in the molecule. This is done by
    checking the number of edges in the molecule and penalizing based on how far
    the molecule is from the desired number of edges.

    :param mol [rdkit.Chem.Mol]: The molecule to compute the penalty for.
    :return [float]: The penalty to apply to the molecule.
    '''

    # check that the mol that is input is
    # a valid molecule
    if mol is None:
        return 0.0
    
    # check that the number of edges is within the desired range
    desired_min_edges = 5 
    desired_max_edges = 15 

    # get the actual number of edges
    actual_edges = mol.GetNumBonds()
    
    # compute the penalty
    epen = 0.0

    # penalize if the number of edges is less than the desired minimum
    # with an L2 norm penalty
    if actual_edges < desired_min_edges:
        epen = ((desired_min_edges - actual_edges) * EDGE_COUNT_PENALTY_SCALING) ** 2

    elif actual_edges > desired_max_edges:
        epen = ((actual_edges - desired_max_edges) * EDGE_COUNT_PENALTY_SCALING) ** 2

    return epen


def compute_size_penalty(mol):
    
    '''
    We are trying to coax the model to generate smaller molecules with this one
    which is done by penalizing the size of the molecule.

    :param mol [rdkit.Chem.Mol]: The molecule to compute the penalty for.
    :return [float]: The penalty to apply to the molecule.
    '''

    # check that the mol that is input is
    # a valid molecule
    if mol is None:
        return 0.0
    
    # get the number of atoms in the molecule
    num_atoms = mol.GetNumAtoms()

    # penalize the size of the molecule
    if num_atoms < 7:
        return ((5 - num_atoms) * SIZE_SCALER) ** 2
    
    elif num_atoms > 20:
        return (num_atoms - 15) * SIZE_SCALER
    
    return 0.0

def compute_valence_penalty(mol):
    
    '''
    Compute the penalty for invalid valences in the molecule. This is done by
    counting the number of invalid valences in the molecule and scaling the
    penalty by the number of atoms in the molecule.

    :param mol [rdkit.Chem.Mol]: The molecule to compute the penalty for.
    :return [float]: The penalty to apply to the molecule
    '''

    # check that the mol that is input is
    # a valid molecule
    if mol is None:
        return 0.0
    
    # get the number of atoms in the molecule
    num_atoms = mol.GetNumAtoms()

    # penalize the valence of the molecule
    if num_atoms <= 0:
        num_atoms = 1
    
    # check the number of valence electrons that should not be there
    num_invalid_valences = count_invalid_valences(mol)
    return num_invalid_valences * VALENCE_PENALTY_SCALING / num_atoms
