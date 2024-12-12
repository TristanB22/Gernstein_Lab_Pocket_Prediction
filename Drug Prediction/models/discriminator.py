# models/discriminator.py

'''
Discriminator model for evaluating the validity of generated molecular graphs.

This module defines the Discriminator class, which assesses the authenticity of
molecular graphs generated by the Generator. The discriminator utilizes residual
Graph Isomorphism Network (GIN) layers followed by global pooling and fully
connected layers to produce a validity score for each molecule.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import ResidualGINLayer
from torch_geometric.nn import global_mean_pool
import logging


class Discriminator(nn.Module):
    '''
    Discriminator model for evaluating generated molecular graphs.

    This class defines the architecture of the discriminator, which takes molecular
    graphs as input and outputs a score indicating their validity. The discriminator
    uses residual Graph Isomorphism Network (GIN) layers to capture graph features,
    followed by global mean pooling and fully connected layers to produce the final
    validity score.

    :param input_dim [int]: Dimension of the input node features.
    :param hidden_dim [int]: Dimension of the hidden layers.
    :param num_layers [int]: Number of residual GIN layers.
    '''
    
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Discriminator, self).__init__()
        logging.info("[discriminator.__init__] initializing discriminator with attention layers.")
        
        # define residual Graph Isomorphism Network layers
        self.layers = nn.ModuleList([
            ResidualGINLayer(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # define global attention mechanism
        self.global_attn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # define pooling layer
        self.pool = global_mean_pool
        
        # define fully connected layers for final score prediction
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        logging.info("[discriminator.__init__] discriminator initialized successfully with attention layers.")
    
    def forward(self, x, edge_index, batch):
        '''
        Forward pass for the Discriminator.

        Processes the input molecular graphs through residual GIN layers, applies
        global pooling, and passes the result through fully connected layers to
        obtain a validity score.

        :param x [torch.Tensor]: Node feature matrix.
        :param edge_index [torch.LongTensor]: Edge indices defining the graph connectivity.
        :param batch [torch.Tensor]: Batch vector assigning each node to a graph in the batch.
        :return [torch.Tensor]: Validity scores for each molecular graph in the batch.
        '''
        
        # pass through residual GIN layers
        for layer in self.layers:
            x = layer(x, edge_index)
        
        # apply global mean pooling to aggregate node features
        x = self.pool(x, batch)
        
        # pass through fully connected layers to obtain final score
        x = self.fc(x)
        
        return x
