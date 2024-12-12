# rl/utils.py

'''
Computing and applying flags to calculate the final reward for a molecule
based on various reward and penalty components.
'''

from config import (
    FLAGS, 
    VALIDITY_REWARD, 
    FOOL_SCALING, 
    UNIQUE_MOL_SCALER, 
    VALENCE_PENALTY_SCALING, 
    SIZE_SCALER
)


def apply_flags_to_reward(base_reward, reward_dict, penalty_dict):
    '''
    Applies various flags to compute the final reward for a molecule based on 
    predefined reward and penalty dictionaries.
    
    :param base_reward [float]: The initial reward value.
    :param reward_dict [dict]: Dictionary containing different reward components.
    :param penalty_dict [dict]: Dictionary containing different penalty components.
    :return [float]: The final computed reward after applying all enabled flags.
    '''
    
    final_reward = 0.0
    
    # enable validity reward
    if FLAGS['ENABLE_VALIDITY_REWARD']:
        validity_score = reward_dict.get('validity_score', 0.0)
        final_reward += validity_score
    
    # enable fool reward
    if FLAGS['ENABLE_FOOL_REWARD']:
        fool_reward_val = reward_dict.get('fool_reward', 0.0)
        final_reward += fool_reward_val
    
    # enable unique molecule penalty
    if FLAGS['ENABLE_UNIQUE_MOL_PENALTY']:
        unique_mol_pen_val = penalty_dict.get('unique_mol_penalty', 0.0)
        final_reward -= unique_mol_pen_val
    
    # enable motif reward
    if FLAGS['ENABLE_MOTIF_REWARD']:
        motif_reward_val = reward_dict.get('motif_reward', 0.0)
        final_reward += motif_reward_val
    
    # enable similarity reward
    if FLAGS['ENABLE_SIMILARITY_REWARD']:
        similarity_val = reward_dict.get('similarity_reward', 0.0)
        final_reward += similarity_val
    
    # enable valence check
    if FLAGS['ENABLE_VALENCE_CHECK']:
        valence_adjustment = reward_dict.get('valence_adjustment_base', 0.0)
        final_reward += valence_adjustment
    
    # enable connectivity penalty
    if FLAGS['ENABLE_CONNECTIVITY_PENALTY']:
        disconnect_penalty = penalty_dict.get('disconnect_penalty', 0.0)
        final_reward -= disconnect_penalty
    
    # enable distribution penalty
    if FLAGS['ENABLE_DISTRIBUTION_PENALTY']:
        distribution_penalty = penalty_dict.get('distribution_penalty', 0.0)
        final_reward -= distribution_penalty
    
    # enable same atom penalty
    if FLAGS['ENABLE_SAME_ATOM_PENALTY']:
        same_atom_penalty = penalty_dict.get('same_atom_penalty', 0.0)
        final_reward -= same_atom_penalty
    
    # enable duplicate molecule penalty
    if FLAGS['ENABLE_DUPLICATE_MOLECULE_PENALTY']:
        duplicate_penalty = penalty_dict.get('duplicate_penalty', 0.0)
        final_reward -= duplicate_penalty
    
    # enable edge density penalty
    if FLAGS['ENABLE_EDGE_DENSITY_PENALTY']:
        edge_density_penalty = penalty_dict.get('edge_density_penalty', 0.0)
        final_reward -= edge_density_penalty
    
    # enable invalid edge penalty
    if FLAGS['ENABLE_INVALID_EDGE_PENALTY']:
        invalid_edge_penalty = penalty_dict.get('invalid_edge_penalty', 0.0)
        final_reward -= invalid_edge_penalty
    
    # enable edge count penalty
    if FLAGS['ENABLE_EDGE_COUNT_PENALTY']:
        edge_count_penalty = penalty_dict.get('edge_count_penalty', 0.0)
        final_reward -= edge_count_penalty
    
    # enable size penalty
    if FLAGS['ENABLE_SIZE_PENALTY']:
        size_penalty = penalty_dict.get('size_penalty', 0.0)
        final_reward -= size_penalty
    
    return final_reward
