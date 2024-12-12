# data/constants.py

'''
This file defines all of the constants that are associated with the molecule generation
program. It includes things like the atom types that we are going to allow for the generation
process and the motifs that can be checked during the reinforcement learning process to reward
molecules that are more likely to be valid than not. 
'''

from rdkit import Chem

# the atom types that we are considering for the molecule generation
allowed_atom_types = ['C', 'O', 'N', 'H', 'S', 'Br', 'P', 'Cl', 'F', 'I']
max_valence_dict = {
    'C': 4, 'N': 3, 'O': 2, 'S': 2, 'F': 1,
    'P': 5, 'Cl': 1, 'Br': 1, 'H': 1, 'I': 1
}

# the motifs that we can compare parts of the molecule against
motif_smarts = [
    Chem.MolFromSmarts("c1ccccc1"),
    Chem.MolFromSmarts("C=O"),
    Chem.MolFromSmarts("N=C=O"),
    Chem.MolFromSmarts("c1ccncc1"),
    Chem.MolFromSmarts("C1=CC=CO1"),
    Chem.MolFromSmarts("C1=CC=CS1"),
    Chem.MolFromSmarts("C1=CN=CN1"),
    Chem.MolFromSmarts("C1=NC=CN1"),
    Chem.MolFromSmarts("C1=CC=NC1"),
    Chem.MolFromSmarts("C1=NC=CC=N1"),
    Chem.MolFromSmarts("O=C(C)Oc1ccccc1"),
    Chem.MolFromSmarts("CC(=O)NC"),
    Chem.MolFromSmarts("C#N"),
    Chem.MolFromSmarts("C(=O)N"),
    Chem.MolFromSmarts("C1=CC=C(O)C=C1"),
    Chem.MolFromSmarts("C1=CC=C(CN)C=C1"),
    Chem.MolFromSmarts("C1CCC(CC1)N"),
    Chem.MolFromSmarts("C1=CN=C2C=CC=CC2=N1"),
    Chem.MolFromSmarts("C1=CC=C2C(C=CC2=O)=C1"),
    Chem.MolFromSmarts("C1=CC=NC2=C1C=CC=C2"),
    Chem.MolFromSmarts("C1CNCCN1"),
    Chem.MolFromSmarts("C1=CC=C(C=C1)O"),
    Chem.MolFromSmarts("C1=CC2=C(C=C1)N=CN2"),
    Chem.MolFromSmarts("C1=CC2=C(C=C1)C=NC=N2"),
]
