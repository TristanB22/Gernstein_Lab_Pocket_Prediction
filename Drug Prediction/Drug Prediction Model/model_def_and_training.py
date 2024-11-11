# this file is going to contain all of the code that 
# we can use for the training and generation of the graphs that describe the molecules
# that we are interested in. We are going to define a message passing neural network
# that can be used to aggregate and move the information that is stored in the graph
# around the molecule.

import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, GlobalAttention
from torch_geometric.data import Data, DataLoader


class MolecularGraphDataset(Dataset):
    
	# a class that contains all of the graphs that define the 
	# molecules we are interested in predicting the effects of
	def __init__(self, root, transform=None, pre_transform=None):
		super(MolecularGraphDataset, self).__init__(root, transform, pre_transform)


	# this function is going to load in the data from the raw files
	# and return the data as a list of Data objects
	def process(self):
		
		# process and save graphs and corresponding pocket features
		data_list = []
		
		# iterate through the paths and load the information in
		for mol_path in self.raw_paths:

			# load in the molecule and pocket data
			mol_graph = self.process_molecule(mol_path)

			# get the pocket features and the output from the original U-Net
			protein_features, pocket_features = self.process_pocket(mol_path.replace('molecule', 'pocket'))

			# append the data to the list
			data_list.append(Data(x=mol_graph.x, edge_index=mol_graph.edge_index, edge_attr=mol_graph.edge_attr, pocket_features=pocket_features, protein_features=protein_features))

		# save the data
		torch.save(self.collate(data_list), self.processed_paths[0])


	def process_pocket(self, pocket_path: str, protein_path: str):
		
		# get the pocket information and protein information
		# that has been precomputed by the U-Net
		pocket_data = torch.load(pocket_path)

		# get the protein voxelized version
		protein_voxelized = torch.load(protein_path)

		# return the pocket data
		return protein_voxelized, pocket_data




class MolecularMPNN(torch.nn.Module):
    
	# message passing neural network that is going to be used to
	# predict the next added segment of a graph that we are constructing that describes
	# a molecule 

	def __init__(self, node_in_feats, edge_in_feats, global_feats, hidden_dim, choices_per_node):
		
		super(MolecularMPNN, self).__init__()
		
		# define the edge network as a sequence of linear transformations with relu activations
		self.edge_network = torch.nn.Sequential(
			torch.nn.Linear(edge_in_feats, hidden_dim),
			torch.nn.ReLU(),
			torch.nn.Linear(hidden_dim, hidden_dim * node_in_feats)
		)
		
		# message passing neural network using the defined edge network and mean aggregation
		self.conv = NNConv(node_in_feats, hidden_dim, self.edge_network, aggr='mean')
		
		# global attention mechanism to integrate information across the graph
		self.global_pool = GlobalAttention(gate_nn=torch.nn.Linear(hidden_dim, 1))
		
		# fully connected layer to combine node and global features
		self.fc1 = torch.nn.Linear(hidden_dim + global_feats, hidden_dim)
		
		# output layer for generating node predictions
		self.node_pred = torch.nn.Linear(hidden_dim, choices_per_node)  
		
		# output layer for the stop signal
		self.stop_pred = torch.nn.Linear(hidden_dim, 1) 


	# the forward pass of the model to predict the next component
	def forward(self, data):
		
		# the edges and forward pass of the model
		x, edge_index, edge_attr, batch, pocket_features = data.x, data.edge_index, data.edge_attr, data.batch, data.pocket_features
		x = self.conv(x, edge_index, edge_attr)
		x = F.relu(x)
		
		# global context pooled across the entire graph
		global_context = self.global_pool(x, batch)
		
		# expand global context and concatenate it to each node feature
		global_context = global_context[batch]  
		node_features = torch.cat([x, global_context, pocket_features[batch]], dim=1)
		
		# process the node features through a forward pass of the model
		node_features = self.fc1(node_features)
		node_features = F.relu(node_features)
		
		# predict the next node actions
		node_predictions = self.node_pred(node_features)
		node_predictions = F.log_softmax(node_predictions, dim=-1)
		
		# predict the stop signal
		stop_signal = self.stop_pred(global_context)
		stop_signal = torch.sigmoid(stop_signal).squeeze(-1)  # apply sigmoid to convert to probability
		
		return node_predictions, stop_signal