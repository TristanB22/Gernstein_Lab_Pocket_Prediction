# models/generator.py

'''
Generator model for molecule generation using residual graph attention layers.

This module defines the Generator class, which is responsible for generating molecular graphs
based on input noise vectors. The generator utilizes residual Graph Attention Convolutional
layers to produce complex and chemically valid molecules.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import ResidualGATConv
from data.constants import max_valence_dict, allowed_atom_types
from data.utils import MolEncoder
from config import TEMPERATURE
from torch_geometric.data import Data
from utils.general import sample_distribution
from data.process import keep_largest_component_pyg

import logging


class Generator(nn.Module):
    '''
    Generator model for generating molecular graphs.

    This class defines the architecture of the generator, which takes random noise
    as input and generates molecular graphs by predicting atom types, bond types,
    and other molecular properties.

    :param noise_dim [int]: Dimension of the input noise vector.
    :param hidden_dim [int]: Dimension of the hidden layers.
    :param num_node_features [int]: Number of features per node (atom).
    :param num_edge_features [int]: Number of features per edge (bond).
    :param max_nodes [int]: Maximum number of nodes (atoms) in a generated molecule.
    :param num_layers [int]: Number of residual graph attention layers.
    '''
    
    def __init__(self, noise_dim, hidden_dim, num_node_features, num_edge_features, max_nodes, num_layers=16):
        super(Generator, self).__init__()
        logging.info("[generator.__init__] initializing generator.")
        
        # initialize model parameters
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.max_nodes = max_nodes

        # register buffer for maximum valences based on atom types
        max_valences = [max_valence_dict.get(t, 4) for t in MolEncoder.atom_types]
        self.register_buffer("max_valences_tensor", torch.tensor(max_valences, dtype=torch.float))

        # define the first fully connected layer
        self.fc1 = nn.Linear(noise_dim, hidden_dim)
        
        # define residual Graph Attention Convolutional layers
        self.convs = nn.ModuleList([ResidualGATConv(hidden_dim, hidden_dim) for _ in range(num_layers)])

        # define fully connected layers for node feature predictions
        self.node_fc_atom_type = nn.Linear(hidden_dim, len(MolEncoder.atom_types))
        self.node_fc_degree = nn.Linear(hidden_dim, 1)
        self.node_fc_formal_charge = nn.Linear(hidden_dim, 1)
        self.node_fc_hybridization = nn.Linear(hidden_dim, 3)
        self.node_fc_is_aromatic = nn.Linear(hidden_dim, 1)

        # define fully connected layers for edge feature predictions
        self.edge_fc_exist = nn.Linear(num_node_features * 2, 1)
        self.edge_fc_type = nn.Linear(num_node_features * 2, num_edge_features)

        # define activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        logging.info("[generator.__init__] generator initialized successfully.")

    def forward(self, batch_size, num_nodes=None):
        '''
        Forward pass for the Generator.

        Generates a batch of molecular graphs based on input noise vectors.

        :param batch_size [int]: Number of molecules to generate in the batch.
        :param num_nodes [int, optional]: Fixed number of nodes for each molecule. If None, randomize.
        :return [tuple]: A tuple containing:
            - data_list [list of Data]: Generated molecular graphs.
            - atom_type_log_probs_batch [torch.Tensor]: Log probabilities for atom type predictions.
            - hybridization_log_probs_batch [torch.Tensor]: Log probabilities for hybridization predictions.
        '''
        
        # determine the number of nodes per molecule
        if num_nodes is None:
            num_nodes_list = torch.randint(5, self.max_nodes + 1, (batch_size,), device=self.max_valences_tensor.device)
        else:
            num_nodes_list = torch.full((batch_size,), num_nodes, dtype=torch.int, device=self.max_valences_tensor.device)

        # sample random noise and pass through the first fully connected layer
        noise = torch.randn(batch_size, self.noise_dim, device=self.max_valences_tensor.device)
        h = self.relu(self.fc1(noise))

        # initialize lists to store generated data and log probabilities
        data_list = []
        all_atom_type_log_probs = []
        all_hybridization_log_probs = []

        # iterate over each molecule in the batch
        for i in range(batch_size):
            current_num_nodes = num_nodes_list[i].item()
            x = h[i].unsqueeze(0).repeat(current_num_nodes, 1)

            # initialize edge indices
            if current_num_nodes > 1:
                # create a linear chain of atoms
                chain_edges = [[j, j + 1] for j in range(current_num_nodes - 1)]
                extra_edges = []
                num_extra = torch.randint(1, min(current_num_nodes, 4), (1,)).item()
                
                # add random extra edges
                for _ in range(num_extra):
                    u, v = torch.randint(0, current_num_nodes, (2,))
                    if u != v:
                        if [u.item(), v.item()] not in chain_edges and [v.item(), u.item()] not in chain_edges:
                            extra_edges.append([u.item(), v.item()])
                
                # optionally add a ring structure
                if current_num_nodes > 4:
                    u_ring = torch.randint(0, current_num_nodes, (1,)).item()
                    v_ring = (u_ring + torch.randint(2, current_num_nodes - 1, (1,)).item()) % current_num_nodes
                    if [u_ring, v_ring] not in chain_edges and [v_ring, u_ring] not in chain_edges and \
                       [u_ring, v_ring] not in extra_edges and [v_ring, u_ring] not in extra_edges:
                        extra_edges.append([u_ring, v_ring])
                
                # combine chain and extra edges
                all_edges = chain_edges + extra_edges
                edge_index = torch.tensor(all_edges, dtype=torch.long, device=self.max_valences_tensor.device).t().contiguous()
                
                # make edges undirected
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            else:
                # no edges for single-atom molecules
                edge_index = torch.empty((2, 0), dtype=torch.long, device=self.max_valences_tensor.device)

            # pass through residual Graph Attention Convolutional layers
            for conv in self.convs:
                if edge_index.size(1) > 0:
                    x = conv(x, edge_index)

            # predict atom types
            atom_type_logits = self.node_fc_atom_type(x)
            atom_type_log_probs = F.log_softmax(atom_type_logits, dim=-1)
            atom_type_probs = F.softmax(atom_type_logits, dim=-1)
            atom_type_indices = sample_distribution(atom_type_probs, temperature=TEMPERATURE)

            # predict hybridization states
            hybridization_logits = self.node_fc_hybridization(x)
            hybridization_log_probs = F.log_softmax(hybridization_logits, dim=-1)
            hybridization_probs = F.softmax(hybridization_logits, dim=-1)
            hybridization_indices = sample_distribution(hybridization_probs, temperature=TEMPERATURE)

            # compute mean log probabilities for the molecule
            per_molecule_atom_type_log_prob = atom_type_log_probs.gather(1, atom_type_indices.unsqueeze(-1)).squeeze(-1).mean()
            per_molecule_hybridization_log_prob = hybridization_log_probs.gather(1, hybridization_indices.unsqueeze(-1)).squeeze(-1).mean()

            # predict aromaticity
            is_aromatic_logits = self.node_fc_is_aromatic(x)
            is_aromatic_probs = torch.sigmoid(is_aromatic_logits)
            is_aromatic_samples = torch.bernoulli(is_aromatic_probs).long().float()

            # one-hot encode atom types
            atom_type_one_hot = torch.zeros_like(atom_type_probs)
            atom_type_one_hot.scatter_(1, atom_type_indices.unsqueeze(-1), 1.0)

            # one-hot encode hybridization states
            hybridization_one_hot = torch.zeros_like(hybridization_probs)
            hybridization_one_hot.scatter_(1, hybridization_indices.unsqueeze(-1), 1.0)

            # determine chosen atom types and normalize valences
            chosen_atom_type_idx = atom_type_indices
            chosen_valences = torch.tensor(
                [max_valence_dict.get(MolEncoder.atom_types[idx.item()], 4) for idx in chosen_atom_type_idx],
                dtype=torch.float, device=x.device
            )
            normalized_valences = chosen_valences / 5.0
            normalized_valences = normalized_valences.unsqueeze(-1)

            # concatenate all node features
            node_features = torch.cat([
                atom_type_one_hot,
                torch.sigmoid(self.node_fc_degree(x)),
                torch.tanh(self.node_fc_formal_charge(x).clone().detach()),
                hybridization_one_hot,
                is_aromatic_samples,
                normalized_valences
            ], dim=1)

            # predict edge existence and types
            if edge_index.size(1) > 0:
                u = edge_index[0, :]
                v = edge_index[1, :]
                edge_features = torch.cat([node_features[u], node_features[v]], dim=1)
                edge_exist_logits = self.edge_fc_exist(edge_features)
                edge_type_logits = self.edge_fc_type(edge_features)

                # determine which edges exist
                edge_exist_probs = self.sigmoid(edge_exist_logits)
                edge_mask = (edge_exist_probs > 0.5).squeeze(-1).bool()

                # sample edge types based on predicted probabilities
                edge_type_probs = self.softmax(edge_type_logits)
                edge_type_indices = sample_distribution(edge_type_probs, temperature=TEMPERATURE)
                edge_attr_one_hot = torch.zeros_like(edge_type_probs)
                edge_attr_one_hot.scatter_(1, edge_type_indices.unsqueeze(-1), 1.0)
                edge_attr_filtered = edge_attr_one_hot[edge_mask]
                edge_index_filtered = edge_index[:, edge_mask]
            else:
                # no edges to process
                edge_index_filtered = torch.empty((2, 0), dtype=torch.long, device=self.max_valences_tensor.device)
                edge_attr_filtered = torch.empty((0, self.num_edge_features), dtype=torch.float, device=self.max_valences_tensor.device)

            # create a PyTorch Geometric Data object for the molecule
            data = Data(
                x=node_features,
                edge_index=edge_index_filtered,
                edge_attr=edge_attr_filtered,
                num_nodes=current_num_nodes
            ).to(self.max_valences_tensor.device)

            # keep only the largest connected component to ensure molecule validity
            data = keep_largest_component_pyg(data)
            if data is not None and data.num_nodes > 0:
                data_list.append(data)
                all_atom_type_log_probs.append(per_molecule_atom_type_log_prob)
                all_hybridization_log_probs.append(per_molecule_hybridization_log_prob)

        # stack log probabilities for the batch
        atom_type_log_probs_batch = torch.stack(all_atom_type_log_probs, dim=0)
        hybridization_log_probs_batch = torch.stack(all_hybridization_log_probs, dim=0)

        return data_list, atom_type_log_probs_batch, hybridization_log_probs_batch
