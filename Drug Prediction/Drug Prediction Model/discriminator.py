# discriminator.py

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
import torch.nn.utils.spectral_norm as spectral_norm

class Discriminator(nn.Module):
    
	"""
    The discriminator network is going to evaluate whether the 
    molecule that is input is real or not  
	"""

	def __init__(self, input_dim, hidden_dim, num_layers):

		super(Discriminator, self).__init__()
		self.convs = nn.ModuleList()
		

		# GINConv layers
		for i in range(num_layers):
			
			# check whether this is the first layer or not
			if i == 0:
				
				# the first layer puts it in the embedding dimension
				nn_layer = nn.Sequential(

					# spectral normalization to stabilize training
					spectral_norm(nn.Linear(input_dim, hidden_dim)),
					nn.ReLU(),
					spectral_norm(nn.Linear(hidden_dim, hidden_dim))
				)
			
			else:
				
				# stay in the hidden dimension for the layers after the first
				nn_layer = nn.Sequential(
					spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
					nn.ReLU(),
					spectral_norm(nn.Linear(hidden_dim, hidden_dim))
				)

			# add the GINConv layer
			conv = GINConv(nn_layer)

			# add the layer to the list of layers
			self.convs.append(conv)
		
		# final prediction for the molecule 
		self.fc = nn.Linear(hidden_dim, 1)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x, edge_index, batch) -> bool:
		"""
		Forward pass for the molecule. This will take in the molecule]
		and return whether it is real or not

		:param x: node features
		:param edge_index: edge index
		:param batch: batch index

		:ret [bool]: whether the molecule is real or not
		"""
		
		# run through each of the layers
		for conv in self.convs:
			x = conv(x, edge_index)
		
		# get the graphical representation
		x = global_mean_pool(x, batch)
		
		# decide whether this molecule is real or not
		x = self.fc(x)
		x = self.sigmoid(x)
		return x
