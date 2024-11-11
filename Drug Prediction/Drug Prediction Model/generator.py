# generator.py

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data

from graph_generation import MolEncoder

class Generator(nn.Module):

	"""
	Network that will generate the molecules that we are looking for
	"""

	def __init__(self, noise_dim, hidden_dim, num_node_features, num_edge_features, max_nodes, num_layers=8):
		
		super(Generator, self).__init__()
		
		# the dimension of the noise vector that we are feeding
		# to increase the variance of the molecules generated
		self.noise_dim = noise_dim

		# dimension of the hidden layer
		self.hidden_dim = hidden_dim

		# number of node features
		self.num_node_features = num_node_features
		
		# maximum number of nodes in the backbone
		self.max_nodes = max_nodes  

		# initial linear layer to transform noise into initial hidden state
		self.fc1 = nn.Linear(noise_dim, hidden_dim)

		# convolution generator with attention
		# to actually make the molecules
		self.conv_layers = nn.ModuleList([
			TransformerConv(
				in_channels=hidden_dim,
				out_channels=hidden_dim,
				heads=4,
				concat=False,  
				dropout=0.1,
				edge_dim=num_edge_features
			)

			# the number of layers in the generative model
			for _ in range(num_layers)
		])

		# output layers to produce node and edge probabilities
		self.node_fc = nn.Linear(hidden_dim, num_node_features)
		self.edge_fc = nn.Linear(hidden_dim * 2, num_edge_features)

		# the atom type and other features layers
		self.node_fc_atom_type = nn.Linear(hidden_dim, len(MolEncoder.atom_types) - 1)  
		self.node_fc_other_features = nn.Linear(hidden_dim, num_node_features - 1)

		# activation functions
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()


	def forward(self, batch_size, num_nodes=None) -> list:

		"""
		Forward pass through the generative network to create the molecule output

		:param batch_size [int]: the number of molecules to generate
		:param num_nodes [int]: the number of nodes in the molecule to generate

		:ret [list]: list of Data objects representing the generated molecules
		"""

		# if the number of nodes is not specified, generate a random number of nodes
		if num_nodes is None:
			num_nodes_list = torch.randint(1, self.max_nodes + 1, (batch_size,))
		else:
			num_nodes_list = torch.full((batch_size,), num_nodes, dtype=torch.int)

		# how many nodes we could generate in this batch
		max_num_nodes = num_nodes_list.max().item()

		# gnerate noise
		device = next(self.parameters()).device  
		noise = torch.randn(batch_size, self.noise_dim, device=device)

		# first state
		h = self.relu(self.fc1(noise))  

		# list to store the generated data
		data_list = []
		for i in range(batch_size):
			
			# number of nodes in the molecule
			num_nodes = num_nodes_list[i].item()

			# start the generation process
			# shape: [num_nodes, hidden_dim]
			x = h[i].unsqueeze(0).repeat(num_nodes, 1)  

			# make the fully connected graph
			node_indices = torch.arange(num_nodes, device=device)
			
			# create the edge index
			edge_index = torch.combinations(node_indices, r=2).t()
			
			# edges will be bidirectional
			edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  

			# initialize edge attributes
			edge_attr = torch.randn(edge_index.size(1), self.num_edge_features, device=device)

			# apply TransformerConv layers
			for conv in self.conv_layers:
				x = self.relu(conv(x, edge_index, edge_attr))

			# predict node features and types for each of the atoms
			# shape: [num_nodes, num_atom_types - 1]
			atom_type_logits = self.node_fc_atom_type(x)
			atom_type_probs = torch.softmax(atom_type_logits, dim=-1)  

			# predict other node features
			# shape: [num_nodes, num_node_features - 1]
			other_features = self.node_fc_other_features(x)
			other_features = self.sigmoid(other_features)

			# combine the features for the final prediction
			node_features = torch.cat([atom_type_probs, other_features], dim=1)
			node_features = self.sigmoid(node_features)  

			# get the edge features for the otutput
			src_nodes = x[edge_index[0]]
			dst_nodes = x[edge_index[1]]
			edge_input = torch.cat([src_nodes, dst_nodes], dim=1)
			edge_features = self.edge_fc(edge_input)
			edge_features = self.sigmoid(edge_features)

			# create Data object
			data = Data(
				x=node_features,
				edge_index=edge_index,
				edge_attr=edge_features
			)

			# get the number of nodes for the object
			data.num_nodes = num_nodes

			# add the data object to the list
			data_list.append(data)

		return data_list
