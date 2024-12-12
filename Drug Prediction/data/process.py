# data/process.py

'''
The process module contains functions for processing molecules and converting them to PyTorch Geometric Data objects.
'''

import os
import torch
import logging
import traceback
import numpy as np
from rdkit import Chem
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
from tqdm import tqdm
from collections import Counter
import pandas as pd
import networkx as nx
from multiprocessing import Pool, cpu_count
from .constants import allowed_atom_types, max_valence_dict
from .utils import MolEncoder
from config import LOGGING_ENABLED, PRINT_ERROR
from rdkit.Chem import AllChem, QED, Draw, Descriptors
from rdkit import Chem



def atom_features(atom):
    
    '''
    Function for extracting atom features from an RDKit atom object.

    :param atom [rdkit.Chem.rdchem.Atom]: RDKit atom object.
    :return features [torch.Tensor]: Tensor of atom features.
    '''

    # one-hot encoding of atom type for the tensor
    atom_type = atom.GetSymbol()
    atom_type_one_hot = [1 if atom_type == t else 0 for t in allowed_atom_types]
    
    # normalizing the degree and charges of the atom
    normalized_degree = atom.GetDegree() / 6.0
    formal_charge_normalized = (atom.GetFormalCharge() - (-1)) / (1 - (-1))
    normalized_formal_charge = torch.tanh(torch.tensor(formal_charge_normalized, dtype=torch.float))
    
    # one-hot representation of hybridization
    hybridization = atom.GetHybridization()
    hybridization_one_hot = [
        1 if hybridization == Chem.rdchem.HybridizationType.SP else 0,
        1 if hybridization == Chem.rdchem.HybridizationType.SP2 else 0,
        1 if hybridization == Chem.rdchem.HybridizationType.SP3 else 0
    ]

    # checking other atomic attributes
    is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    valence = max_valence_dict.get(atom_type, 4)
    normalized_valence = valence / 5.0
    features = [
        *atom_type_one_hot,
        normalized_degree,
        normalized_formal_charge,
        *hybridization_one_hot,
        is_aromatic,
        normalized_valence
    ]

    # return the attributes in tensor form
    return torch.tensor(features, dtype=torch.float)



def bond_features(bond):
    
    '''
    Get the bond features from an RDKit bond object.

    :param bond [rdkit.Chem.rdchem.Bond]: RDKit bond object.
    :return features [torch.Tensor]: Tensor of bond features.
    '''

    # one-hot encoding of bond type
    bt = bond.GetBondType()

    # one-hot representation of bond type
    bond_type_one_hot = [
        1 if bt == Chem.rdchem.BondType.SINGLE else 0,
        1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bt == Chem.rdchem.BondType.AROMATIC else 0
    ]

    # checking if the bond is conjugated or in a ring
    # and placing info into tensor
    is_conjugated = 1.0 if bond.GetIsConjugated() else 0.0
    is_in_ring = 1.0 if bond.IsInRing() else 0.0
    
    # return the bond features in tensor form
    return torch.tensor([*bond_type_one_hot, is_conjugated, is_in_ring], dtype=torch.float)



def graph_to_pyg_data(G):
    
    '''
    Transform a networkx graph object to a PyTorch Geometric Data object.

    :param G [networkx.Graph]: NetworkX graph object.
    :return data [torch_geometric.data.Data]: PyTorch Geometric Data object.
    '''

    # convert the networkx graph to a pt geo data obj
    data = from_networkx(G)

    # get the node features
    node_features = [G.nodes[n]['features'] for n in G.nodes()]
    data.x = torch.stack(node_features, dim=0)
    
    # get the edge features by iterating through the edges
    edge_features = []
    for u,v,edge_attr in G.edges(data=True):
        edge_features.append(edge_attr['features'])
        edge_features.append(edge_attr['features'])

    # stack the edge features into a tensor
    if edge_features:
        data.edge_attr = torch.stack(edge_features, dim=0)
    else:
        data.edge_attr = torch.empty((0, 6), dtype=torch.float, device=data.x.device)

    # return the data object
    return data


def get_backbone_graph(mol):
    
    '''
    Get the backbone graph of a molecule.

    :param mol [rdkit.Chem.rdchem.Mol]: RDKit molecule object.
    :return G [networkx.Graph]: NetworkX graph object.
    '''

    # the graph object that we are going to fill with bond and atom features
    G = nx.Graph()
    backbone_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in allowed_atom_types]

    # if there are no backbone atoms, return None
    if len(backbone_atoms) == 0:
        return None
    
    # add the atoms to the graph
    for atom_idx in backbone_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        G.add_node(atom_idx, features=atom_features(atom))

    # add the bonds to the graph
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if u in backbone_atoms and v in backbone_atoms:
            G.add_edge(u, v, features=bond_features(bond))

    # return the graph object
    # so long as we added something to the graph
    if G.number_of_nodes() > 0:
        return G
    else:
        return None


def keep_largest_component_pyg(data):
    
    '''
    Get the largest connected component of a PyTorch Geometric Data object
    as part of the postprocessing scheme.

    :param data [torch_geometric.data.Data]: PyTorch Geometric Data object.
    :return data [torch_geometric.data.Data]: PyTorch Geometric Data object.
    '''

    # get the edge index from the data object
    edge_index = data.edge_index.cpu().numpy()
    
    # the graph that we are going to add info to
    G = nx.Graph()

    # add the nodes to the graph
    for i in range(data.num_nodes):
        G.add_node(i)

    # add the edges to the graph
    for j in range(edge_index.shape[1]):
        u = edge_index[0, j]
        v = edge_index[1, j]
        G.add_edge(u, v)

    # get the connected components
    components = list(nx.connected_components(G))
    if len(components) == 0:
        return None
    
    # get the largest component
    # and return that as the data object
    largest_comp = max(components, key=len)
    largest_comp = sorted(list(largest_comp))

    # mask of the data object information
    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    
    # set the mask to true for the largest component
    mask[largest_comp] = True
    data = data.subgraph(mask)

    # return the data object
    return data if data.num_nodes > 0 else None



def process_single_sdf(sdf_file):
    
    '''
    Get a single SDF file's contained ligands for procesisng.

    :param sdf_file [str]: Path to the SDF file.
    :return data_list [list]: List of PyTorch Geometric Data objects.
    '''

    # the list of data objects that we are going to return
    data_list = []

    # the supplier object for the SDF file
    supplier = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)

    # check each of the molecules in the SDF file
    for idx, mol in enumerate(supplier):
        
        # if the molecule is None, skip it
        if mol is None:
            continue
        
        # try to sanitize the molecule
        try:
            Chem.SanitizeMol(mol)
        except:
            continue
        
        # get the backbone graph
        G = get_backbone_graph(mol)
        
        # make sure the graph is not None
        if G is None:
            continue
        
        # try to convert the graph to a pt geo dat object
        try:
            data = graph_to_pyg_data(G)
        except Exception:
            continue
        
        # keep the largest component of the data object
        data = keep_largest_component_pyg(data)
        if data is not None and data.num_nodes > 0:
            data_list.append(data)
        
    return data_list


def collect_sdf_files(root_dir, max_files=None):
    
    '''
    Getting all of the SDF files in a directory.

    :param root_dir [str]: Root directory to search for SDF files.
    :param max_files [int]: Maximum number of files to collect.
    :return files [list]: List of SDF files.
    '''

    # keeping track of the files
    files=[]

    # walk through the directory
    # and check each of the files in each subdirectory
    for subdir,_,f in os.walk(root_dir):
        
        # check each of the files
        for file in f:
            
            # check if this is one of the files that we are after
            if file.endswith('.sdf'):
                
                # add the file to the list
                files.append(os.path.join(subdir,file))
                if max_files and len(files)>=max_files:
                    return files
    return files



def process_molecules_multiprocessing(root_dir, max_files=None):
    
    '''
    Get the molecules from the SDF files in a directory using multiprocessing.

    :param root_dir [str]: Root directory to search for SDF files.
    :param max_files [int]: Maximum number of files to collect.
    :return all_data [list]: List of PyTorch Geometric Data objects.
    '''

    from config import LOGGING_ENABLED

    # print out logging info to the file
    logging.info("[process_molecules_multiprocessing] starting multiprocessing for molecule processing.")
    
    # get all of the sdf files that we care about
    sdf_files = collect_sdf_files(root_dir, max_files)

    # print out the number of files that we are going to process
    if LOGGING_ENABLED:
        logging.info(f"found {len(sdf_files)} .sdf files to process.")

    # the list of data objects that we are going to return
    all_data = []
    try:
        with Pool(max(cpu_count() // 2, 1)) as pool:
            for result in tqdm(pool.imap(process_single_sdf, sdf_files), total=len(sdf_files), desc="processing sdf files"):
                all_data.extend(result)
    except Exception as e:
        logging.error(f"[process_molecules_multiprocessing] error during multiprocessing: {e}")
        traceback.print_exc()

    # log that we are done
    logging.info(f"[process_molecules_multiprocessing] completed processing. total valid molecules: {len(all_data)}")
    return all_data




def MOSES_smiles_to_data(smiles):
    
    '''
    For the MOSES dataset, we want to convert SMILES strings to pt geo data objects.

    :param smiles [str]: SMILES string.
    :return data [torch_geometric.data.Data]: PyTorch Geometric Data object.
    '''


    # use the molencoder from utils
    from .utils import MolEncoder
    
    from_networkx = from_networkx

    # get the molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # sanitize the molecule
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
    
    # get the backbone graph
    G = MOSES_get_backbone_graph(mol)
    
    # check if the output graph is legit or not
    if G is None:
        return None

    # try to convert the graph to a pt geo data object
    try:
        data = MOSES_graph_to_pyg_data(G)
    except Exception:
        return None

    # keep the largest component of the data object
    data = MOSES_keep_largest_component_pyg(data)
    if data is not None and data.num_nodes > 0:
        return data
    
    return None



def MOSES_atom_features(atom):
    
    '''
    Process the atom features of the molecules that we get from the MOSES dataset

    :param atom [rdkit.Chem.rdchem.Atom]: RDKit atom object.
    :return features [torch.Tensor]: Tensor of atom features.
    '''

    # one-hot encoding of atom type for the tensor
    atom_type = atom.GetSymbol()
    atom_type_one_hot = [1 if atom_type == t else 0 for t in allowed_atom_types]
    
    # normalizing the degree and charges of the atom
    normalized_degree = atom.GetDegree() / 6.0
    formal_charge_normalized = (atom.GetFormalCharge() - (-1)) / (1 - (-1))
    normalized_formal_charge = torch.tanh(torch.tensor(formal_charge_normalized, dtype=torch.float))
    
    # one-hot representation of hybridization
    hybridization = atom.GetHybridization()
    hybridization_one_hot = [
        1 if hybridization == Chem.rdchem.HybridizationType.SP else 0,
        1 if hybridization == Chem.rdchem.HybridizationType.SP2 else 0,
        1 if hybridization == Chem.rdchem.HybridizationType.SP3 else 0
    ]

    # checking other atomic attributes
    is_aromatic = 1.0 if atom.GetIsAromatic() else 0.0
    valence = max_valence_dict.get(atom_type, 4)
    normalized_valence = valence / 5.0
    features = [
        *atom_type_one_hot,
        normalized_degree,
        normalized_formal_charge,
        *hybridization_one_hot,
        is_aromatic,
        normalized_valence
    ]
    return torch.tensor(features, dtype=torch.float)




def MOSES_bond_features(bond):
        '''
        Get the bond features from an RDKit bond object for the MOSES dataset.

        :param bond [rdkit.Chem.rdchem.Bond]: RDKit bond object.
        :return features [torch.Tensor]: Tensor of bond features.
        '''

        # get the bond type
        bt = bond.GetBondType()

        # one-hot encode the bond type
        bond_type_one_hot = [
            1 if bt == Chem.rdchem.BondType.SINGLE else 0,
            1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
            1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
            1 if bt == Chem.rdchem.BondType.AROMATIC else 0
        ]

        # check if the bond is conjugated
        is_conjugated = 1.0 if bond.GetIsConjugated() else 0.0

        # check if the bond is in a ring
        is_in_ring = 1.0 if bond.IsInRing() else 0.0

        # return the features as a tensor
        return torch.tensor([*bond_type_one_hot, is_conjugated, is_in_ring], dtype=torch.float)




def MOSES_get_backbone_graph(mol):
    '''
    Get the backbone graph of a molecule for the MOSES dataset.

    :param mol [rdkit.Chem.rdchem.Mol]: RDKit molecule object.
    :return G [networkx.Graph]: NetworkX graph object.
    '''

    # create an empty graph
    G = nx.Graph()

    # get the indices of backbone atoms
    backbone_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in allowed_atom_types]

    # if there are no backbone atoms, return None
    if len(backbone_atoms) == 0:
        return None

    # add the atoms to the graph
    for atom_idx in backbone_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        G.add_node(atom_idx, features=MOSES_atom_features(atom))

    # add the bonds to the graph
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if u in backbone_atoms and v in backbone_atoms:
            G.add_edge(u, v, features=MOSES_bond_features(bond))

    # return the graph object if it has nodes
    if G.number_of_nodes() > 0:
        return G
    else:
        return None




def MOSES_graph_to_pyg_data(G):
    '''
    Converts a NetworkX graph to a PyTorch Geometric data object.

    :param G [networkx.Graph]: The input graph with node and edge features.
    :return data [torch_geometric.data.Data]: The converted data object with node features (data.x) 
                                              and edge features (data.edge_attr).
    '''

    # convert the networkx graph to a pytorch geometric data object
    data = from_networkx(G)

    # extract node features from the graph and stack them into a tensor
    node_features = [G.nodes[n]['features'] for n in G.nodes()]
    data.x = torch.stack(node_features, dim=0)

    # extract edge features from the graph and stack them into a tensor
    edge_features = []
    for u, v, edge_attr in G.edges(data=True):
        edge_features.append(edge_attr['features'])
        edge_features.append(edge_attr['features'])

    # if there are edge features, stack them into a tensor, otherwise create an empty tensor
    if edge_features:
        data.edge_attr = torch.stack(edge_features, dim=0)
    else:
        data.edge_attr = torch.empty((0, 6), dtype=torch.float, device=data.x.device)

    # return the pytorch geometric data object
    return data




def MOSES_keep_largest_component_pyg(data):

    '''
    Get the largest connected component of a pt geo data object for the MOSES dataset.

    :param data [torch_geometric.data.Data]: PyTorch Geometric Data object.
    :return data [torch_geometric.data.Data]: PyTorch Geometric Data object.
    '''

    import networkx as nx

    # get the edge index from the data object
    edge_index = data.edge_index.cpu().numpy()
    
    # create the graph that we are going to add information to 
    G = nx.Graph()

    # add the nodes to the graph
    for i in range(data.num_nodes):
        G.add_node(i)

    # add the edges to the graph
    for j in range(edge_index.shape[1]):
        u = edge_index[0, j]
        v = edge_index[1, j]
        G.add_edge(u, v)

    # get the connected components
    components = list(nx.connected_components(G))
    
    # make sure that componetns in the graph even exist
    if len(components) == 0:
        return None
    
    # get the largest component in the molecule
    largest_comp = max(components, key=len)
    largest_comp = sorted(list(largest_comp))

    # mask of the data object information
    mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=data.x.device)
    mask[largest_comp] = True

    # return the data object
    data = data.subgraph(mask)
    return data if data.num_nodes > 0 else None



def MOSES_load_moses_data(csv_path):

    '''
    Load the MOSES dataset from a CSV file.

    :param csv_path [str]: Path to the CSV file.
    :return data_list [list]: List of PyTorch Geometric Data objects.
    '''

    # load in the csv file with the smiles string information
    df = pd.read_csv(csv_path)

    # get the smiles strings from the dataframe
    smiles_list = df['SMILES'].dropna().tolist()


    # convert the smiles strings to data objects
    data_list = []

    # iterate through the smiles strings
    for sm in smiles_list:
        
        # parse the smiles string and convert it to a data object that is appropriate
        data = MOSES_smiles_to_data(sm)
        if data is not None:
            data_list.append(data)

    return data_list



def keep_largest_fragment_mol(mol):

    '''
    Keep the largest fragment of a molecule.

    :param mol [rdkit.Chem.rdchem.Mol]: RDKit molecule object.
    :return largest_frag [rdkit.Chem.rdchem.Mol]: Largest fragment of the molecule.
    '''


    # try to get the largest fragment of the molecule
    try:

        # get the fragments of the molecule
        mol = Chem.Mol(mol)

        # get the largest fragment
        frags = Chem.GetMolFrags(mol, asMols=True, sanitize=False)

        # if there are no fragments, return None
        if len(frags) == 0:
            logging.warning("No fragments found during fragmentation.")
            return None
        
        # get the largest fragment based on the number of atoms in the molecule
        largest_frag = max(frags, key=lambda m: m.GetNumAtoms())
        Chem.SanitizeMol(largest_frag)
        logging.info("Largest fragment sanitized successfully.")
        return largest_frag

    # catch any exceptions that occur
    except Chem.rdchem.AtomValenceException as e:
        logging.error(f"Sanitization failed for molecule fragment: {e}")
        return None
    
    # catch any other exceptions that occur
    except Exception as e:
        logging.error(f"Error in keep_largest_fragment_mol: {e}")
        return None





def data_to_molecule(data, allow_invalid=True):

    '''
    Fucntion that is able to take the edge and node representation of the graph and
    convert it back into a molecule object.

    :param data [torch_geometric.data.Data]: PyTorch Geometric Data object.
    :param allow_invalid [bool]: Whether or not to allow invalid molecules.
    :return mol [rdkit.Chem.rdchem.Mol]: RDKit molecule object.
    '''


    # make the imports that we need
    from .constants import allowed_atom_types, max_valence_dict
    
    # check that there is data that we have processed
    if data is None:
        return None,0
    
    try:
    
        # create a new RDKit molecule object
        mol = Chem.RWMol()

        # get the node features and edge information
        # for processing
        atom_types = allowed_atom_types
        node_features = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        
        # keep track of how many edges the data says that we need to add that we do nto 
        # actually have to add
        invalid_edge_count = 0

        # add the atoms to the molecule
        for i in range(node_features.size(0)):

            # get the atom type and add it to the molecule
            atom_type_idx = torch.argmax(node_features[i][:len(atom_types)]).item()
            atom_type = atom_types[atom_type_idx] if atom_type_idx < len(atom_types) else 'C'
            atom = Chem.Atom(atom_type)

            # add the atom features to the molecule
            formal_charge = node_features[i][11].item()
            
            # set the formal charge of the atom
            atom.SetFormalCharge(int(round(formal_charge)))
            
            # set the hybridization of the atom
            hybridization_one_hot = node_features[i][12:15]

            # set the hybridization of the atom
            if hybridization_one_hot[0].item() > 0.5:
                atom.SetHybridization(Chem.rdchem.HybridizationType.SP)
            elif hybridization_one_hot[1].item() > 0.5:
                atom.SetHybridization(Chem.rdchem.HybridizationType.SP2)
            elif hybridization_one_hot[2].item() > 0.5:
                atom.SetHybridization(Chem.rdchem.HybridizationType.SP3)

            # set the aromaticity of the atom
            # based on the node features
            is_aromatic = node_features[i][15].item()
            
            # set the aromaticity of the atom
            if is_aromatic > 0.5:
                atom.SetIsAromatic(True)

            # add the atom to the molecule
            mol.AddAtom(atom)

        # parse the edges now
        for i in range(edge_index.size(1)):
            
            # get the edge information
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()

            # check if the edge is valid
            if u >= node_features.size(0) or v >= node_features.size(0) or u == v:
                invalid_edge_count += 1
                continue

            # check if the bond already exists
            if mol.GetBondBetweenAtoms(u, v) is not None:
                continue

            # get the bond type and make sure that the size is not zero
            if edge_attr.size(0) == 0:
                bond_type = Chem.rdchem.BondType.SINGLE

            else:

                # get the bond type and add it to the molecule
                bond_attr_idx = i % edge_attr.size(0)
                bond_type_probs = edge_attr[bond_attr_idx][:4]
                bond_type_idx = torch.argmax(bond_type_probs).item()
                bond_types = [
                    Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC
                ]

                # add the bond to the molecule
                bond_type = bond_types[bond_type_idx] if bond_type_idx < len(bond_types) else Chem.rdchem.BondType.SINGLE

            # add the bond to the molecule
            try:
                mol.AddBond(u, v, bond_type)
            except:
                invalid_edge_count += 1

        # convert the molecule to a molecule object
        mol.UpdatePropertyCache(strict=False)
        
        # check if the molecule is valid
        # and whether the user cares or not that it is
        if not allow_invalid:
            try:
                Chem.SanitizeMol(mol)
            except:
                return None, invalid_edge_count
        return mol, invalid_edge_count
    except:
        return None, 0
