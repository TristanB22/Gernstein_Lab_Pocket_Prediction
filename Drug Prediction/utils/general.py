# utils/general.py

'''
General utility functions for the project.

This module provides utility functions that are used across different parts of the project,
including functions for sampling from probability distributions.
'''

import torch
import torch.nn.functional as F


def sample_distribution(dist, temperature=1.0):
    '''
    Sample indices from a probability distribution with optional temperature scaling.
    
    :param dist [torch.Tensor]: The input probability distribution tensor.
    :param temperature [float]: The temperature parameter for scaling the distribution. Default is 1.0.
    :return [torch.Tensor]: The sampled indices from the distribution.
    '''
    
    # clamp the distribution to avoid numerical issues
    dist = dist.clamp(min=1e-9)
    
    # apply temperature scaling if temperature is not 1.0
    if temperature != 1.0:
        dist = dist.pow(1.0 / temperature)
    
    # normalize the distribution
    dist_sum = dist.sum(dim=-1, keepdim=True)
    dist = dist / torch.max(dist_sum, torch.tensor(1e-9, device=dist.device))
    
    # handle cases where distribution contains NaNs or Infs
    if torch.isnan(dist).any() or torch.isinf(dist).any():
        dist = torch.ones_like(dist)
        dist = dist / dist.sum(dim=-1, keepdim=True)
    
    # sample indices based on the distribution
    sampled_indices = torch.multinomial(dist, 1).squeeze(-1)
    
    return sampled_indices
