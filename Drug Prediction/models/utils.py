# models/utils.py

'''
Utility modules for the generator and discriminator models, including
residual graph convolutional layers and baseline reward tracking.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv


class ResidualGATConv(nn.Module):
    '''
    A residual Graph Attention Convolutional layer.
    This layer applies a GAT convolution followed by a residual connection
    and a ReLU activation.
    
    :param in_channels [int]: Number of input features per node.
    :param out_channels [int]: Number of output features per node.
    '''
    
    def __init__(self, in_channels, out_channels):
        super(ResidualGATConv, self).__init__()
        self.conv = GATConv(in_channels, out_channels, heads=4, concat=False, dropout=0.1)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
    
    def forward(self, x, edge_index):
        '''
        Forward pass for the ResidualGATConv layer.
        
        :param x [torch.Tensor]: Node feature matrix.
        :param edge_index [torch.LongTensor]: Graph connectivity.
        :return [torch.Tensor]: Updated node feature matrix after convolution and residual connection.
        '''
        
        # compute residual connection
        x_res = self.lin(x)
        
        # apply GAT convolution
        x_conv = self.conv(x, edge_index)
        
        # add residual and apply ReLU activation
        return F.relu(x_conv + x_res)


class ResidualGINLayer(nn.Module):
    '''
    A residual Graph Isomorphism Network (GIN) layer. This layer applies a GIN convolution followed by a residual connection
    and a Leaky ReLU activation.
    
    :param in_dim [int]: Number of input features per node.
    :param hidden_dim [int]: Number of hidden features.
    '''
    
    def __init__(self, in_dim, hidden_dim):
        super(ResidualGINLayer, self).__init__()
        nn_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINConv(nn_layer)
        self.lin = nn.Linear(in_dim, hidden_dim, bias=False)
    
    def forward(self, x, edge_index):
        '''
        Forward pass for the ResidualGINLayer.
        
        :param x [torch.Tensor]: Node feature matrix.
        :param edge_index [torch.LongTensor]: Graph connectivity.
        :return [torch.Tensor]: Updated node feature matrix after convolution and residual connection.
        '''
        
        # compute residual connection
        x_res = self.lin(x)
        
        # apply GIN convolution
        x_conv = self.conv(x, edge_index)
        
        # add residual and apply Leaky ReLU activation
        return F.leaky_relu(x_conv + x_res, negative_slope=0.2)


class Baseline:
    '''
    A baseline tracker for reward computation.
    This class maintains a moving average of rewards to stabilize training.
    
    :param alpha [float]: Smoothing factor for the moving average.
    '''
    
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.value = None
    
    def update(self, reward):
        '''
        Updates the baseline with a new reward value.
        
        :param reward [float]: The new reward to incorporate.
        :return [float]: The updated baseline value.
        '''
        if self.value is None:
            self.value = reward
        else:
            
            # print intermediate values for debugging
            print(f"self.alpha: {self.alpha}")
            print(f"self.value: {self.value}")
            print(f"reward: {reward}")
            
            # update the moving average
            self.value = self.alpha * self.value + (1 - self.alpha) * reward
            
        return self.value
