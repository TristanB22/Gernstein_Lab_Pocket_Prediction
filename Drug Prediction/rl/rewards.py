# rl/rewards.py

'''
Computing the rewards that should be applied to a molecule 
based on its properties.
'''

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
from collections import Counter
from data.constants import max_valence_dict, motif_smarts
from config import VALIDITY_REWARD, FOOL_SCALING


def compute_validity_score(mol):
    '''
    Computes the validity score for a molecule based on various chemical properties.
    
    :param mol [rdkit.Chem.Mol]: The molecule to compute the score for.
    :return [float]: The normalized validity score.
    '''
    
    # check that the mol is legit
    if mol is None:
        return 0.0
    
    # keeping track of the score and the number of atoms
    score = 0.0
    total_atoms = mol.GetNumAtoms()
    total_bonds = mol.GetNumBonds()

    # valence check
    for atom in mol.GetAtoms():
        
        # get the valence of the atom and what 
        # the max value of the valence actually should be
        atom_type = atom.GetSymbol()
        max_valence = max_valence_dict.get(atom_type, 4)
        actual_valence = atom.GetExplicitValence() + atom.GetImplicitValence()

        # get the delta in the max valence and the actual valence
        valence_violation = actual_valence - max_valence

        # apply a penalty if the valence is too high
        if valence_violation > 0:
            score -= valence_violation * 0.5  
        elif valence_violation < 0:
            score -= abs(valence_violation) * 0.1

    # bond type validation
    for bond in mol.GetBonds():
        
        # check the bond types and make sure that there are no invalid bonds
        bond_type = bond.GetBondType()
        if bond_type not in [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                             Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]:
            score -= 1.0
            
        # check whether this is an aromatic bond or not
        if bond_type == Chem.rdchem.BondType.AROMATIC:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            
            # remove the score 
            if not (begin_atom.GetIsAromatic() and end_atom.GetIsAromatic()):
                score -= 0.5

    # connectivity check
    g = nx.Graph()
    
    # add the atoms to the graph
    for atom in mol.GetAtoms():
        g.add_node(atom.GetIdx())
        
    # place the bonds in the graph
    for bond in mol.GetBonds():
        g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        
    # compute the number of components for the graph
    num_components = nx.number_connected_components(g)

    # make sure that there are not too many components in the graph
    if num_components > 1:
        score -= (num_components - 1) * 1.5
        largest_fragment_size = max(len(c) for c in nx.connected_components(g))
        score += (largest_fragment_size / total_atoms) * 0.5
    else:
        score += 1.0

    # formal charge
    total_formal_charge = Chem.GetFormalCharge(mol)
    if abs(total_formal_charge) > 1:
        score -= abs(total_formal_charge) * 0.3

    # chirality
    for atom in mol.GetAtoms():
        if atom.HasProp('_ChiralityPossible') and atom.GetChiralTag() == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
            score -= 0.2

    # implicit hydrogens
    for atom in mol.GetAtoms():
        implicit_h = atom.GetNumImplicitHs()
        if implicit_h < 0:
            score -= 0.3

    # compute the normalized penalty so that this scales with the number of bonds that you have
    max_possible_penalty = (total_atoms * 1.0) + (total_bonds * 1.0) + 3.0
    normalized_score = (score + max_possible_penalty) / (2 * max_possible_penalty)
    normalized_score = max(0.0, min(normalized_score, 1.0))
    return normalized_score


def strict_valence_check(mol):
    '''
    Checks if the molecule strictly adheres to valence rules.
    
    :param mol [rdkit.Chem.Mol]: The molecule to check.
    :return [bool]: True if valence rules are strictly followed, False otherwise.
    '''
    
    # check that the mol is legit
    if mol is None:
        return False
    
    # check valence for each atom
    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        max_val = max_valence_dict.get(atom_type, 4)
        val = atom.GetExplicitValence() + atom.GetImplicitValence()
        if val > max_val:
            return False
    return True


def compute_motif_reward(mol):
    '''
    Computes the motif-based reward for a molecule based on the presence of specific substructures.
    
    :param mol [rdkit.Chem.Mol]: The molecule to compute the reward for.
    :return [float]: The reward to apply to the molecule.
    '''
    
    # check that the mol is legit
    if mol is None:
        return 0.0
    
    reward = 0.0
    for motif in motif_smarts:
        if motif is not None and mol.HasSubstructMatch(motif):
            reward += 0.5
    return reward


def compute_similarity_reward(mol, ref_fps, threshold=0.4):
    '''
    Computes the similarity reward based on Tanimoto similarity to reference fingerprints.
    
    :param mol [rdkit.Chem.Mol]: The molecule to compute the reward for.
    :param ref_fps [list]: List of reference fingerprints.
    :param threshold [float]: Similarity threshold to earn a reward.
    :return [float]: The reward to apply to the molecule.
    '''
    
    # check that the mol is legit and reference fingerprints exist
    if mol is None or not ref_fps:
        return 0.0
    
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        similarities = [Chem.DataStructs.TanimotoSimilarity(fp, rfp) for rfp in ref_fps]
        max_sim = max(similarities) if similarities else 0
        if max_sim > threshold:
            return (max_sim - threshold) * 2.0
        else:
            return 0.0
    except:
        return 0.0
