# main.py

'''
Main execution script for training the molecular generative model.

This script initializes the generator and discriminator models, loads data,
configures logging, and manages the training loop including pretraining and
reinforcement learning (RL) phases. It also handles metrics computation,
logging, and visualization of generated molecules.

Run:
    python3 main.py --dataset moses
or
    python3 main.py --dataset pdbbind
'''

import os
import json
import time
import traceback
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.data import Batch
from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict
import random
import argparse

from config import (DEVICE, TEMPERATURE, SEED, TRAIN_ON_SMALL_DATASET, SMALL_DAT_NUM_MOLS,
                    LOGGING_ENABLED, run_dir, loss_file_path, gen_disc_loss_path, 
                    EPOCHS, PRETRAIN_EPOCHS, RL_BATCH_SIZE, REINFORCEMENT_LEARNING_FACTOR, 
                    ENABLE_REINFORCEMENT_LEARNING, PRINT_INFO, LOAD_MODEL, FLAGS, 
                    VALIDITY_REWARD, FOOL_SCALING, UNIQUE_MOL_SCALER, SIZE_SCALER,
                    WRITE_MOLECULE_IMAGES, VISUALIZE_EVERY_INSTANCE, VISUALIZE_MOLECULE_EVERY_EPOCH, LOG_RL_LOSSES,
                    PDBBIND_LIGAND_SDF_DIR, moses_csv_path)

from data.process import (MOSES_load_moses_data, data_to_molecule, process_molecules_multiprocessing, keep_largest_component_pyg)
from data.constants import allowed_atom_types
from models.generator import Generator
from models.discriminator import Discriminator
from models.utils import Baseline
from utils.plotting import plot_atom_type_distribution, visualize_molecule
from utils.general import sample_distribution
from rl.penalties import (compute_disconnect_penalty, compute_distribution_penalty,
                          compute_same_atom_penalty, compute_duplicate_penalty,
                          compute_edge_density_penalty, compute_invalid_edge_penalty,
                          compute_edge_count_penalty, compute_size_penalty, compute_valence_penalty)
from rl.rewards import compute_validity_score, compute_motif_reward, compute_similarity_reward
from rl.utils import apply_flags_to_reward
from rdkit import Chem
from rdkit.Chem import AllChem, QED, Draw, Descriptors
import scipy.stats as stats

from torch_geometric.loader import DataLoader


def initialize_models(valid_data):
    '''
    Initialize the generator, discriminator, and baseline models.

    :param valid_data [list]: A list of valid molecular data objects.
    :return [tuple]: A tuple containing the generator, discriminator, baseline models,
                    device, number of node features, number of edge features,
                    batch size, number of epochs, and learning rate.
    '''
    from config import NOISE_DIM, HIDDEN_DIM, NUM_LAYERS, MAX_NODES, LEARNING_RATE
    logging.info("[initialize_models] initializing models.")
    
    # check for valid molecules
    if len(valid_data) == 0:
        raise ValueError("No valid molecules.")
    
    NUM_NODE_FEATURES = 17
    assert valid_data[0].x.size(1) == 17, "Expected 17 node features."
    
    # determine number of edge features
    if valid_data[0].edge_attr.size(0) > 0:
        NUM_EDGE_FEATURES = valid_data[0].edge_attr.size(1)
    else:
        NUM_EDGE_FEATURES = 6
    assert NUM_EDGE_FEATURES == 6, "Expected 6 edge features."
    
    device = DEVICE
    logging.info(f"Using device: {device}")
    
    # initialize baseline model
    baseline = Baseline(alpha=0.9)
    from data.utils import MolEncoder
    
    # initialize generator model
    generator = Generator(
        noise_dim=NOISE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_node_features=NUM_NODE_FEATURES,
        num_edge_features=NUM_EDGE_FEATURES,
        max_nodes=MAX_NODES,
        num_layers=NUM_LAYERS
    ).to(device)
    
    # initialize discriminator model
    discriminator = Discriminator(
        input_dim=NUM_NODE_FEATURES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    
    logging.info("Models initialized.")
    return generator, discriminator, baseline, device, NUM_NODE_FEATURES, NUM_EDGE_FEATURES, \
           128, EPOCHS, LEARNING_RATE


def load_existing_models(generator, discriminator, load_best, load_checkpoint_epoch=None):
    '''
    Load existing generator and discriminator models from checkpoints.

    :param generator [Generator]: The generator model instance.
    :param discriminator [Discriminator]: The discriminator model instance.
    :param load_best [bool]: Flag indicating whether to load the best models.
    :param load_checkpoint_epoch [int, optional]: Specific epoch checkpoint to load.
    '''
    if load_best:
        try:
            generator.load_state_dict(torch.load('best_generator.pth'))
            discriminator.load_state_dict(torch.load('best_discriminator.pth'))
            logging.info("Loaded best models.")
        except FileNotFoundError:
            logging.warning("Best models not found.")
    elif load_checkpoint_epoch is not None:
        try:
            generator.load_state_dict(torch.load(f'generator_epoch_{load_checkpoint_epoch}.pth'))
            discriminator.load_state_dict(torch.load(f'discriminator_epoch_{load_checkpoint_epoch}.pth'))
            logging.info(f"Loaded models from epoch {load_checkpoint_epoch}.")
        except FileNotFoundError:
            logging.warning("Checkpoint models not found.")
    else:
        logging.info("No model loading requested.")


def compute_diversity(smiles_list):
    '''
    Compute the diversity score of a list of SMILES strings based on Tanimoto similarity.

    :param smiles_list [list]: A list of SMILES strings representing molecules.
    :return [float]: The diversity score, where 1 indicates maximum diversity.
    '''
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    
    # return maximum diversity for small lists
    if len(smiles_list) < 2:
        return 1.0
    
    # generate fingerprints for each molecule
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2) for s in smiles_list if Chem.MolFromSmiles(s)]
    
    # return maximum diversity if insufficient valid fingerprints
    if len(fps) < 2:
        return 1.0
    
    sims = []
    # compute pairwise Tanimoto similarities
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sims.append(sim)
    
    # return 1 minus the average similarity
    if len(sims) == 0:
        return 1.0
    return 1 - np.mean(sims)


def compute_rl_metrics(rl_losses):
    '''
    Compute average reinforcement learning metrics from loss data.

    :param rl_losses [list]: A list of dictionaries containing RL loss metrics.
    :return [dict]: A dictionary with average metrics.
    '''
    if not rl_losses:
        return {}
    avg_rl_metrics = {k: sum(d[k] for d in rl_losses) / len(rl_losses) for k in rl_losses[0].keys()}
    return avg_rl_metrics


def update_metrics(metrics, atom_count, validity_score, fool_reward, unique_mol_penalty,
                   disconn_pen, kldiv_pen, satom_pen, dup_pen, edge_pen, invalid_edge_penalty,
                   edge_count_penalty, size_penalty, valence_penalty, motif_reward,
                   similarity_reward, reward_total):
    '''
    Update the metrics dictionary with new data for a specific atom count.

    :param metrics [dict]: The metrics dictionary to update.
    :param atom_count [int]: The number of atoms in the molecule.
    :param validity_score [float]: The validity score of the molecule.
    :param fool_reward [float]: The fool reward value.
    :param unique_mol_penalty [float]: The unique molecule penalty.
    :param disconn_pen [float]: The disconnect penalty.
    :param kldiv_pen [float]: The distribution penalty.
    :param satom_pen [float]: The same atom penalty.
    :param dup_pen [float]: The duplicate penalty.
    :param edge_pen [float]: The edge density penalty.
    :param invalid_edge_penalty [float]: The invalid edge penalty.
    :param edge_count_penalty [float]: The edge count penalty.
    :param size_penalty [float]: The size penalty.
    :param valence_penalty [float]: The valence penalty.
    :param motif_reward [float]: The motif reward.
    :param similarity_reward [float]: The similarity reward.
    :param reward_total [float]: The total RL reward.
    '''
    if atom_count > 0:
        atom_metrics = metrics['per_atom_metrics'][atom_count]
        atom_metrics['validity_scores'].append(validity_score)
        atom_metrics['fool_rewards'].append(fool_reward)
        atom_metrics['unique_mol_penalties'].append(unique_mol_penalty)
        atom_metrics['disconnect_penalties'].append(disconn_pen)
        atom_metrics['distribution_penalties'].append(kldiv_pen)
        atom_metrics['same_atom_penalties'].append(satom_pen)
        atom_metrics['duplicate_penalties'].append(dup_pen)
        atom_metrics['edge_density_penalties'].append(edge_pen)
        atom_metrics['invalid_edge_penalties'].append(invalid_edge_penalty)
        atom_metrics['edge_count_penalties'].append(edge_count_penalty)
        atom_metrics['size_penalties'].append(size_penalty)
        atom_metrics['valence_penalties'].append(valence_penalty)
        atom_metrics['motif_rewards'].append(motif_reward)
        atom_metrics['similarity_rewards'].append(similarity_reward)
        atom_metrics['total_rl_rewards'].append(reward_total)


def compute_rewards(generated_molecules, z_scores):
    '''
    Compute the rewards and penalties for generated molecules based on various criteria.

    :param generated_molecules [list]: A list of RDKit molecule objects.
    :param z_scores [numpy.ndarray]: Z-scores corresponding to each molecule's validity.
    :return [tuple]: A tuple containing lists of total rewards, individual rewards, penalties,
                    validity scores, unique molecule penalties, motif rewards, and similarity rewards.
    '''
    from rl.penalties import (compute_disconnect_penalty, compute_distribution_penalty,
                              compute_same_atom_penalty, compute_duplicate_penalty,
                              compute_edge_density_penalty, compute_invalid_edge_penalty,
                              compute_edge_count_penalty, compute_size_penalty, compute_valence_penalty)
    from rl.rewards import (compute_validity_score, compute_motif_reward, compute_similarity_reward)
    from config import FLAGS, VALIDITY_REWARD, FOOL_SCALING, UNIQUE_MOL_SCALER

    rewards_total = []
    rewards = []
    penalties = []
    validity_scores = []
    unique_mol_penalties = []
    motif_rewards_list = []
    similarity_rewards_list = []

    train_fps = []
    for mol, z_score in zip(generated_molecules, z_scores):
        reward_total = 0.0
        reward_dict = {}
        penalty_dict = {}

        # compute validity reward
        if FLAGS['ENABLE_VALIDITY_REWARD']:
            validity_score = compute_validity_score(mol)
            rew = validity_score * VALIDITY_REWARD
            reward_dict['validity_score'] = rew
            reward_total += rew
        else:
            validity_score = compute_validity_score(mol)

        # compute fool reward
        if FLAGS['ENABLE_FOOL_REWARD']:
            rew = float(z_score * FOOL_SCALING)
            reward_dict['fool_reward'] = rew
            reward_total += rew
        else:
            reward_dict['fool_reward'] = 0.0

        # compute unique molecule penalty
        if FLAGS['ENABLE_UNIQUE_MOL_PENALTY']:
            unique_smiles = set([Chem.MolToSmiles(m) for m in generated_molecules if m is not None])
            unique_mol_penalty = float((len(unique_smiles) - 1) * 1.0 * UNIQUE_MOL_SCALER) if len(unique_smiles) > 1 else 0.0
            penalty_dict['unique_mol_penalty'] = unique_mol_penalty
            reward_total -= unique_mol_penalty
        else:
            penalty_dict['unique_mol_penalty'] = 0.0

        # compute disconnect penalty
        if FLAGS['ENABLE_DISCONNECT_PENALTY']:
            dp = compute_disconnect_penalty(mol)
            penalty_dict['disconnect_penalty'] = dp
            reward_total -= dp
        else:
            penalty_dict['disconnect_penalty'] = 0.0

        # compute distribution penalty
        if FLAGS['ENABLE_DISTRIBUTION_PENALTY']:
            dp = compute_distribution_penalty(mol)
            penalty_dict['distribution_penalty'] = dp
            reward_total -= dp
        else:
            penalty_dict['distribution_penalty'] = 0.0

        # compute same atom penalty
        if FLAGS['ENABLE_SAME_ATOM_PENALTY']:
            sp = compute_same_atom_penalty(mol)
            penalty_dict['same_atom_penalty'] = sp
            reward_total -= sp
        else:
            penalty_dict['same_atom_penalty'] = 0.0

        # compute duplicate molecule penalty
        if FLAGS['ENABLE_DUPLICATE_MOLECULE_PENALTY']:
            dpp = compute_duplicate_penalty(mol)
            penalty_dict['duplicate_penalty'] = dpp
            reward_total -= dpp
        else:
            penalty_dict['duplicate_penalty'] = 0.0

        # compute edge density penalty
        if FLAGS['ENABLE_EDGE_DENSITY_PENALTY']:
            ep = compute_edge_density_penalty(mol)
            penalty_dict['edge_density_penalty'] = ep
            reward_total -= ep
        else:
            penalty_dict['edge_density_penalty'] = 0.0

        # compute invalid edge penalty
        if FLAGS['ENABLE_INVALID_EDGE_PENALTY']:
            ip = compute_invalid_edge_penalty(mol)
            penalty_dict['invalid_edge_penalty'] = ip
            reward_total -= ip
        else:
            penalty_dict['invalid_edge_penalty'] = 0.0

        # compute edge count penalty
        if FLAGS['ENABLE_EDGE_COUNT_PENALTY']:
            ecp = compute_edge_count_penalty(mol)
            penalty_dict['edge_count_penalty'] = ecp
            reward_total -= ecp
        else:
            penalty_dict['edge_count_penalty'] = 0.0

        # compute size penalty
        if FLAGS['ENABLE_SIZE_PENALTY']:
            szp = compute_size_penalty(mol)
            penalty_dict['size_penalty'] = szp
            reward_total -= szp
        else:
            penalty_dict['size_penalty'] = 0.0

        # compute valence penalty
        if FLAGS['ENABLE_VALENCE_PENALTY']:
            vp = compute_valence_penalty(mol)
            penalty_dict['valence_penalty'] = vp
            reward_total -= vp
        else:
            penalty_dict['valence_penalty'] = 0.0

        # compute motif reward
        if FLAGS['ENABLE_MOTIF_REWARD']:
            mr = compute_motif_reward(mol)
            reward_dict['motif_reward'] = mr
            reward_total += mr
        else:
            reward_dict['motif_reward'] = 0.0

        # compute similarity reward
        if FLAGS['ENABLE_SIMILARITY_REWARD']:
            sr = compute_similarity_reward(mol, train_fps, threshold=0.4)
            reward_dict['similarity_reward'] = sr
            reward_total += sr
        else:
            reward_dict['similarity_reward'] = 0.0

        # append rewards and penalties to respective lists
        rewards_total.append(reward_total)
        rewards.append(reward_dict)
        penalties.append(penalty_dict)
        validity_scores.append(validity_score)
        unique_mol_penalties.append(penalty_dict['unique_mol_penalty'])
        motif_rewards_list.append(reward_dict.get('motif_reward', 0.0))
        similarity_rewards_list.append(reward_dict.get('similarity_reward', 0.0))

    return rewards_total, rewards, penalties, validity_scores, unique_mol_penalties, motif_rewards_list, similarity_rewards_list


if __name__ == "__main__":
    '''
    Entry point for the training script.
    
    Parses command-line arguments, loads the appropriate dataset, initializes models,
    and starts the training loop with pretraining and reinforcement learning phases.
    Handles logging, metrics computation, and visualization based on configuration flags.
    '''
    # add argument parsing for dataset choice
    parser = argparse.ArgumentParser(description="Train a molecular generative model.")
    parser.add_argument("--dataset", choices=["pdbbind", "moses"], required=True,
                        help="Specify which dataset to use: 'pdbbind' or 'moses'")
    args = parser.parse_args()
    
    from data.process import MOSES_load_moses_data
    from config import (TRAIN_ON_SMALL_DATASET, SMALL_DAT_NUM_MOLS, LOGGING_ENABLED, run_dir,
                        loss_file_path, gen_disc_loss_path, RL_BATCH_SIZE, ENABLE_REINFORCEMENT_LEARNING,
                        PRETRAIN_EPOCHS, WRITE_MOLECULE_IMAGES, VISUALIZE_MOLECULE_EVERY_EPOCH,
                        PRINT_INFO, EPOCHS, FLAGS, LOAD_MODEL, GENERATOR_CHECKPOINT_PATH,
                        moses_csv_path, PDBBIND_LIGAND_SDF_DIR)

    # load data depending on the dataset choice
    if args.dataset == "moses":
        # load MOSES dataset
        valid_data = MOSES_load_moses_data(moses_csv_path)
        if len(valid_data) == 0:
            raise ValueError("No valid molecules processed from MOSES dataset.")
    else:
        # load PDBbind dataset
        max_files = SMALL_DAT_NUM_MOLS if TRAIN_ON_SMALL_DATASET else None
        valid_data = process_molecules_multiprocessing(PDBBIND_LIGAND_SDF_DIR, max_files=max_files)
        # filter valid data
        filtered_data = []
        for idx, d in enumerate(valid_data):
            if hasattr(d, 'x') and d.x is not None and d.x.size(0) > 0 \
               and hasattr(d, 'edge_index') and d.edge_index is not None \
               and hasattr(d, 'edge_attr') and d.edge_attr is not None:
                filtered_data.append(d)
        valid_data = filtered_data
        if len(valid_data) == 0:
            raise ValueError("No valid molecules processed from PDBbind dataset.")

    # initialize data loader
    loader = DataLoader(valid_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    
    # compute average edge density
    average_edge_density = sum(d.edge_index.size(1) for d in valid_data) / \
                           sum(d.num_nodes * (d.num_nodes - 1) for d in valid_data if d.num_nodes > 1) \
                           if len(valid_data) > 0 else 0.0
    
    # count atom types
    atom_counter = Counter()
    for d in valid_data:
        types = torch.argmax(d.x[:, :len(allowed_atom_types)], dim=1).tolist()
        symbols = [allowed_atom_types[t] for t in types]
        atom_counter.update(symbols)
    atom_type_distribution = dict(atom_counter)
    
    # log initial metrics if logging is enabled
    if LOGGING_ENABLED:
        logging.info(f"Using dataset: {args.dataset}")
        logging.info(f"Avg edge density: {average_edge_density}")
        logging.info(f"Atom type distribution: {atom_type_distribution}")

    # initialize models and related parameters
    generator, discriminator, baseline, device, NUM_NODE_FEATURES, NUM_EDGE_FEATURES, BATCH_SIZE, EPOCHS, LEARNING_RATE = initialize_models(valid_data)

    # load existing models if specified
    if LOAD_MODEL:
        load_existing_models(generator, discriminator, load_best=False, load_checkpoint_epoch=10)

    # initialize scaler and optimizers
    scaler = GradScaler()
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # initialize training flags and metrics
    train_discriminator = True
    best_total_loss = float('inf')
    rl_total_loss = []
    per_atom_metrics_json_path = os.path.join(run_dir, "per_atom_metrics.json")
    if not os.path.exists(per_atom_metrics_json_path):
        with open(per_atom_metrics_json_path, "w") as f:
            json.dump([], f, indent=4)

    # initialize metrics dictionary
    metrics = {
        'avg_loss_D': [],
        'avg_loss_G': [],
        'rl_total_reward': [],
        'validity_percentage': [],
        'connectivity_percentage': [],
        'atom_type_distribution': [],
        'bond_type_distribution': [],
        'avg_num_atoms': [],
        'avg_num_bonds': [],
        'novelty_percentage': [],
        'diversity_score': [],
        'avg_molecular_weight': [],
        'avg_logP': [],
        'avg_QED': [],
        'invalid_edge_penalty': [],
        'edge_count_penalty': [],
        'size_penalty': [],
        'valence_penalty': [],
        'avg_motif_reward': [],
        'avg_similarity_reward': [],
        'epoch_time': [],
        'per_atom_metrics': defaultdict(lambda: {
            'validity_scores': [],
            'fool_rewards': [],
            'unique_mol_penalties': [],
            'disconnect_penalties': [],
            'distribution_penalties': [],
            'same_atom_penalties': [],
            'duplicate_penalties': [],
            'edge_density_penalties': [],
            'invalid_edge_penalties': [],
            'edge_count_penalties': [],
            'size_penalties': [],
            'valence_penalties': [],
            'motif_rewards': [],
            'similarity_rewards': [],
            'total_rl_rewards': []
        })
    }

    start_time = time.time()
    train_fps = []
    
    # prepare fingerprints for similarity calculations
    train_mols_for_sim = valid_data[:50]
    for d in train_mols_for_sim:
        rmol, _ = data_to_molecule(d, allow_invalid=True)
        if rmol is not None:
            try:
                Chem.SanitizeMol(rmol)
                fp = AllChem.GetMorganFingerprintAsBitVect(rmol, 2, nBits=2048)
                train_fps.append(fp)
            except:
                pass

    # start training loop
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        logging.info(f"Starting epoch {epoch + 1}/{EPOCHS}")
        generator.train()
        discriminator.train()

        # pretraining phase
        total_loss_D = 0
        total_loss_G = 0
        generated_distributions_epoch = Counter()

        if PRETRAIN_EPOCHS >= epoch:
            for i, real_data in enumerate(tqdm(loader, desc=f"Pretraining pass epoch {epoch+1}/{EPOCHS}")):
                generator_update_freq = 4 + i // 30
                real_data = real_data.to(device)
                batch_size = real_data.num_graphs
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)

                # train discriminator
                if train_discriminator:
                    discriminator.zero_grad()
                    with autocast():
                        # compute discriminator output on real data
                        outputs_real = discriminator(real_data.x, real_data.edge_index, real_data.batch)
                        loss_real = criterion(outputs_real, real_labels)

                        # generate fake data using the generator
                        with torch.no_grad():
                            data_list, atom_type_log_prob, hybridization_log_prob = generator(batch_size)

                        # create a batch from generated data and compute discriminator output
                        fake_batch = Batch.from_data_list(data_list).to(device)
                        outputs_fake = discriminator(fake_batch.x, fake_batch.edge_index, fake_batch.batch)
                        loss_fake = criterion(outputs_fake, fake_labels)

                        # compute total discriminator loss
                        loss_D = loss_real + loss_fake

                    # backpropagate if loss is significant
                    if loss_D.item() > 0.5:
                        scaler.scale(loss_D).backward()
                        clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                        scaler.step(optimizer_D)
                        scaler.update()

                        total_loss_D += loss_D.item()
                        logging.info(f"Discriminator trained with loss: {loss_D.item():.4f}")
                    else:
                        logging.info(f"Discriminator skipped training with loss: {loss_D.item():.4f}")

                # train generator
                generator.zero_grad()
                with autocast():
                    # generate fake data and compute discriminator output
                    data_list, atom_type_log_prob, hybridization_log_prob = generator(batch_size)
                    fake_batch = Batch.from_data_list(data_list).to(device)
                    outputs = discriminator(fake_batch.x, fake_batch.edge_index, fake_batch.batch)
                    
                    # compute generator loss based on discriminator's output
                    loss_G_adv = criterion(outputs, real_labels)
                    loss_G = loss_G_adv

                # skip if loss contains NaN or Inf
                if torch.isnan(loss_G).any() or torch.isinf(loss_G).any():
                    logging.warning(f"Encountered NaN or Inf in generator loss. Skipping backward pass.")
                    continue

                # backpropagate generator loss
                scaler.scale(loss_G).backward()
                clip_grad_norm_(generator.parameters(), max_norm=1.0)
                scaler.step(optimizer_G)
                scaler.update()
                total_loss_G += loss_G.item()

        # compute average losses for the epoch
        avg_loss_D = total_loss_D / len(loader) if len(loader) > 0 else 0
        avg_loss_G = total_loss_G / len(loader) if len(loader) > 0 else 0
        total_loss = avg_loss_D + avg_loss_G
        logging.info(f"[Epoch {epoch + 1}] (Pretraining) Loss D: {avg_loss_D:.4f}, Loss G: {-avg_loss_G:.4f}, Total Loss: {total_loss:.4f}, Training Disc: {train_discriminator}")

        # reinforcement learning phase
        rl_losses = []
        rl_loss_fake = []
        rl_distribution_counter = Counter()
        epoch_valid_smiles = []
        epoch_generated_mols = []

        if ENABLE_REINFORCEMENT_LEARNING and (epoch >= PRETRAIN_EPOCHS):
            for _ in tqdm(range(REINFORCEMENT_LEARNING_FACTOR), desc=f"RL pass epoch {epoch+1}/{EPOCHS}"):
                generator.zero_grad()
                generated_data_list, atom_type_log_prob, hybridization_log_prob = generator(RL_BATCH_SIZE)
                
                # check if generator returned data
                if not isinstance(generated_data_list, list) or len(generated_data_list) == 0:
                    logging.warning("Generator returned an empty data list.")
                    continue
                
                # create a batch from generated data
                fake_batch = Batch.from_data_list(generated_data_list).to(device)
                for mol_data in fake_batch.to_data_list():
                    mol_obj, invalid_edge_count = data_to_molecule(mol_data, allow_invalid=True)
                    epoch_generated_mols.append(mol_obj)
                    smiles = Chem.MolToSmiles(mol_obj) if mol_obj is not None else None
                    epoch_valid_smiles.append(smiles)

                # compute discriminator scores and probabilities
                with torch.no_grad():
                    discriminator_scores = discriminator(fake_batch.x, fake_batch.edge_index, fake_batch.batch)
                    probabilities = torch.sigmoid(discriminator_scores).cpu().numpy()

                # compute z-scores from probabilities
                percentiles = stats.rankdata(probabilities, method='average') / len(probabilities) * 100
                epsilon = 1e-3
                percentiles = np.clip(percentiles, epsilon, 100 - epsilon)
                z_scores = stats.norm.ppf(percentiles / 100)
                z_scores = np.clip(z_scores, -3, 3)

                # collect generated molecules
                generated_molecules = []
                for data_mol in fake_batch.to_data_list():
                    mol, _ = data_to_molecule(data_mol, allow_invalid=True)
                    generated_molecules.append(mol)

                # compute rewards and penalties
                rewards_total, rewards, penalties, validity_scores, unique_mol_penalties, motif_rewards_vals, similarity_rewards_vals = compute_rewards(generated_molecules, z_scores)

                # compute advantages for policy gradient
                mean_reward = np.mean(rewards_total)
                advantages = torch.tensor(rewards_total, dtype=torch.float, device=atom_type_log_prob.device) - torch.tensor(mean_reward, dtype=torch.float, device=atom_type_log_prob.device)
                if advantages.shape[0] != atom_type_log_prob.shape[0]:
                    logging.error(f"Shape mismatch: advantages shape {advantages.shape} vs log_probs shape {atom_type_log_prob.shape}")
                    continue

                # compute RL loss
                rl_losses_tensor = - (atom_type_log_prob + hybridization_log_prob) * advantages
                rl_loss = rl_losses_tensor.mean()

                # backpropagate RL loss
                rl_loss.backward()
                clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()

                # optionally train discriminator with fool reward
                if FLAGS['ENABLE_FOOL_REWARD']:
                    discriminator.zero_grad()
                    fake_labels = torch.zeros(fake_batch.num_graphs, 1, device=device)
                    outputs_fake = discriminator(fake_batch.x.detach(), fake_batch.edge_index.detach(), fake_batch.batch.detach())
                    loss_fake = F.binary_cross_entropy_with_logits(outputs_fake, fake_labels)
                    loss_fake.backward()
                    clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    optimizer_D.step()
                    rl_loss_fake.append(loss_fake.item())

                # update metrics for each molecule
                for i in range(RL_BATCH_SIZE):
                    rl_losses.append({
                        'validity_score': float(validity_scores[i]),
                        'fool_reward': float(z_scores[i]),
                        'unique_mol_penalty': float(unique_mol_penalties[i]),
                        'disconnect_penalty': float(penalties[i].get('disconnect_penalty', 0.0)),
                        'distribution_penalty': float(penalties[i].get('distribution_penalty', 0.0)),
                        'same_atom_penalty': float(penalties[i].get('same_atom_penalty', 0.0)),
                        'duplicate_penalty': float(penalties[i].get('duplicate_penalty', 0.0)),
                        'edge_density_penalty': float(penalties[i].get('edge_density_penalty', 0.0)),
                        'invalid_edge_penalty': float(penalties[i].get('invalid_edge_penalty', 0.0)),
                        'edge_count_penalty': float(penalties[i].get('edge_count_penalty', 0.0)),
                        'size_penalty': float(penalties[i].get('size_penalty', 0.0)),
                        'valence_penalty': float(penalties[i].get('valence_penalty', 0.0)),
                        'motif_reward': float(motif_rewards_vals[i]),
                        'similarity_reward': float(similarity_rewards_vals[i]),
                        'total_rl_reward': float(rewards_total[i])
                    })

                # visualize each generated molecule if enabled
                if VISUALIZE_EVERY_INSTANCE:
                    for i, (reward, trewlist, penalty) in enumerate(zip(rewards_total, rewards, penalties)):
                        print(f"Reward: {reward}\nRewards: {json.dumps(trewlist, indent=4)}\n Penalty: {json.dumps(penalty, indent=4)}")
                        if epoch_generated_mols:
                            mol_obj = epoch_generated_mols[i]
                            try:
                                visualize_molecule(mol_obj, epoch, run_dir, title=f"Generated Molecule at Epoch {epoch + 1}")
                            except Exception as e:
                                logging.error(f"Failed to visualize molecule at epoch {epoch +1}: {e}")
                                traceback.print_exc()
                        else:
                            logging.warning(f"No valid molecules generated in epoch {epoch +1} to visualize.")

                # optionally train discriminator with fool reward
                if FLAGS['ENABLE_FOOL_REWARD']:
                    fake_labels = torch.zeros(fake_batch.num_graphs, 1, device=device)
                    outputs_fake = discriminator(fake_batch.x.detach(), fake_batch.edge_index.detach(), fake_batch.batch.detach())
                    loss_fake = F.binary_cross_entropy_with_logits(outputs_fake, fake_labels)
                    loss_D = loss_fake
                    discriminator.zero_grad()
                    loss_D.backward()
                    clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    optimizer_D.step()
                    total_loss_D += loss_D.item()
                    logging.info(f"RL Discriminator trained with loss: {loss_D.item():.4f}")

        # compute metrics after RL pass
        if ENABLE_REINFORCEMENT_LEARNING and (epoch + 1 > PRETRAIN_EPOCHS) and rl_losses:
            validity_percentage = np.mean([rl['validity_score'] for rl in rl_losses]) if rl_losses else 0
            connectivity_percentage = np.mean([rl['disconnect_penalty'] for rl in rl_losses]) if rl_losses else 0

            # count atom types in generated molecules
            atom_type_counts = Counter()
            for mol in epoch_generated_mols:
                if mol is None:
                    continue
                for atom in mol.GetAtoms():
                    atom_type_counts[atom.GetSymbol()] += 1

            # count bond types in generated molecules
            bond_type_counts = Counter()
            for mol in epoch_generated_mols:
                if mol is None:
                    continue
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()
                    bond_type_counts[str(bond_type)] += 1

            # compute average number of atoms and bonds
            num_atoms_list = [mol.GetNumAtoms() for mol in epoch_generated_mols if mol is not None]
            avg_num_atoms = np.mean(num_atoms_list) if num_atoms_list else 0
            num_bonds_list = [mol.GetNumBonds() for mol in epoch_generated_mols if mol is not None]
            avg_num_bonds = np.mean(num_bonds_list) if num_bonds_list else 0

            # compute novelty and diversity
            unique_generated_smiles = set(s for s in epoch_valid_smiles if s is not None)
            novelty_percentage = (len(unique_generated_smiles) / len(epoch_valid_smiles)) * 100 if len(epoch_valid_smiles) > 0 else 0
            diversity_score = compute_diversity(list(unique_generated_smiles))

            # compute molecular properties
            mol_weights = []
            logP_values = []
            qed_scores = []
            for mol in epoch_generated_mols:
                if mol is None:
                    continue
                try:
                    sanitized_mol = Chem.Mol(mol)
                    Chem.SanitizeMol(sanitized_mol)
                except:
                    sanitized_mol = mol
                try:
                    mw = Descriptors.MolWt(sanitized_mol)
                    mol_weights.append(mw)
                except:
                    pass
                try:
                    logp = Descriptors.MolLogP(sanitized_mol)
                    logP_values.append(logp)
                except:
                    pass
                try:
                    qed_val = QED.qed(sanitized_mol)
                    qed_scores.append(qed_val)
                except:
                    pass

            avg_molecular_weight = np.mean(mol_weights) if mol_weights else 0
            avg_logP = np.mean(logP_values) if logP_values else 0
            avg_QED = np.mean(qed_scores) if qed_scores else 0

            # compute average penalties and rewards
            avg_invalid_edge_penalty = np.mean([rl['invalid_edge_penalty'] for rl in rl_losses]) if rl_losses else 0
            avg_edge_count_penalty = np.mean([rl['edge_count_penalty'] for rl in rl_losses]) if rl_losses else 0
            avg_size_penalty = np.mean([rl['size_penalty'] for rl in rl_losses]) if rl_losses else 0
            avg_valence_penalty = np.mean([rl['valence_penalty'] for rl in rl_losses]) if rl_losses else 0
            avg_motif_reward = np.mean([rl['motif_reward'] for rl in rl_losses]) if rl_losses else 0
            avg_similarity_reward = np.mean([rl['similarity_reward'] for rl in rl_losses]) if rl_losses else 0

            # update metrics dictionary
            metrics['avg_loss_D'].append(avg_loss_D)
            metrics['avg_loss_G'].append(avg_loss_G)
            metrics['rl_total_reward'].append(np.mean([rl['total_rl_reward'] for rl in rl_losses]) if rl_losses else 0)
            metrics['validity_percentage'].append(validity_percentage)
            metrics['connectivity_percentage'].append(connectivity_percentage)
            metrics['atom_type_distribution'].append(atom_type_counts)
            metrics['bond_type_distribution'].append(bond_type_counts)
            metrics['avg_num_atoms'].append(avg_num_atoms)
            metrics['avg_num_bonds'].append(avg_num_bonds)
            metrics['novelty_percentage'].append(novelty_percentage)
            metrics['diversity_score'].append(diversity_score)
            metrics['avg_molecular_weight'].append(avg_molecular_weight)
            metrics['avg_logP'].append(avg_logP)
            metrics['avg_QED'].append(avg_QED)
            metrics['invalid_edge_penalty'].append(avg_invalid_edge_penalty)
            metrics['edge_count_penalty'].append(avg_edge_count_penalty)
            metrics['size_penalty'].append(avg_size_penalty)
            metrics['valence_penalty'].append(avg_valence_penalty)
            metrics['avg_motif_reward'].append(avg_motif_reward)
            metrics['avg_similarity_reward'].append(avg_similarity_reward)

            # compute epoch time
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            metrics['epoch_time'].append(epoch_time)

            # log RL losses if enabled
            if LOG_RL_LOSSES and rl_losses:
                avg_rl = compute_rl_metrics(rl_losses)
                rl_logger = logging.getLogger('rl_logger')
                rl_logger.info(f"Epoch {epoch + 1}: {avg_rl}")

            # append metrics to loss file
            with open(loss_file_path, "a") as f:
                rl_total = metrics['rl_total_reward'][-1] if len(metrics['rl_total_reward']) > 0 else 0
                validity = metrics['validity_percentage'][-1] if metrics['validity_percentage'] else 0
                connectivity = metrics['connectivity_percentage'][-1] if metrics['connectivity_percentage'] else 0
                avg_atoms = metrics['avg_num_atoms'][-1] if metrics['avg_num_atoms'] else 0
                avg_bonds = metrics['avg_num_bonds'][-1] if metrics['avg_num_bonds'] else 0
                novelty = metrics['novelty_percentage'][-1] if metrics['novelty_percentage'] else 0
                diversity = metrics['diversity_score'][-1] if metrics['diversity_score'] else 0
                mw = metrics['avg_molecular_weight'][-1] if metrics['avg_molecular_weight'] else 0
                logp = metrics['avg_logP'][-1] if metrics['avg_logP'] else 0
                qed = metrics['avg_QED'][-1] if metrics['avg_QED'] else 0
                inv_edge_pen = metrics['invalid_edge_penalty'][-1] if metrics['invalid_edge_penalty'] else 0
                edge_count_pen = metrics['edge_count_penalty'][-1] if metrics['edge_count_penalty'] else 0
                size_pen = metrics['size_penalty'][-1] if metrics['size_penalty'] else 0
                valence_pen = metrics['valence_penalty'][-1] if metrics['valence_penalty'] else 0
                motif_reward = metrics['avg_motif_reward'][-1] if metrics['avg_motif_reward'] else 0
                similarity_reward = metrics['avg_similarity_reward'][-1] if metrics['avg_similarity_reward'] else 0

                f.write(f"{epoch+1},{avg_loss_D},{avg_loss_G},{rl_total},{validity},{connectivity},"
                        f"{avg_atoms},{avg_bonds},{novelty},{diversity},{mw},{logp},{qed},"
                        f"{inv_edge_pen},{edge_count_pen},{size_pen},{valence_pen},"
                        f"{motif_reward},{similarity_reward}\n")

            # save latest generator and discriminator models
            try:
                torch.save(generator.state_dict(), os.path.join(run_dir, 'generator_latest.pth'))
                torch.save(discriminator.state_dict(), os.path.join(run_dir, 'discriminator_latest.pth'))
                logging.info(f"[Epoch {epoch +1}] Saved latest models to {run_dir}.")
            except Exception as e:
                logging.error(f"[Epoch {epoch +1}] Failed to save latest models: {e}")
                traceback.print_exc()

        # visualize generated molecule every epoch if enabled
        if VISUALIZE_MOLECULE_EVERY_EPOCH:
            try:
                generator.eval()
                with torch.no_grad():
                    data_list, atom_type_log_prob, hybridization_log_prob = generator(batch_size=1)
                fake_batch = Batch.from_data_list(data_list).to(device)
                mol, invalid_edge_count = data_to_molecule(fake_batch, allow_invalid=True)
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol)
                    logging.info(f"[Epoch {epoch +1}] Generated molecule SMILES: {smiles}")
                    if PRINT_INFO:
                        print(f"Generated molecule SMILES at epoch {epoch +1}: {smiles}")

                visualize_molecule(mol, epoch, run_dir, title=f"Generated Molecule at Epoch {epoch +1}")

                final_distribution = rl_distribution_counter if FLAGS['ENABLE_VALIDITY_REWARD'] and (epoch +1 > PRETRAIN_EPOCHS) else generated_distributions_epoch
                if PRINT_INFO:
                    print(f"final_distribution: {final_distribution}")

                if ENABLE_REINFORCEMENT_LEARNING and (epoch +1 > PRETRAIN_EPOCHS) and rl_losses:
                    avg_rl = compute_rl_metrics(rl_losses)
                    if PRINT_INFO:
                        print(f"RL losses at epoch {epoch +1}:")
                        for k, v in avg_rl.items():
                            print(f"{k}: {v:.4f}")
                plot_atom_type_distribution(final_distribution, title=f"Generated Atom Distribution at Epoch {epoch +1}")
            except Exception as e:
                logging.error(f"[Epoch {epoch +1}] Error visualizing molecule: {e}")
                traceback.print_exc()

        # save per-atom metrics if reinforcement learning is enabled
        if ENABLE_REINFORCEMENT_LEARNING and (epoch +1 > PRETRAIN_EPOCHS):
            per_atom_metrics_json_path = os.path.join(run_dir, "per_atom_metrics.json")
            try:
                with open(per_atom_metrics_json_path, "r") as f:
                    data_json = json.load(f)
            except FileNotFoundError:
                data_json = []
            except Exception as e:
                logging.error(f"Failed to load per-atom metrics file: {e}")
                traceback.print_exc()
                data_json = []

            epoch_metrics_json = {
                'epoch': epoch +1,
                'atom_counts': {}
            }

            # compute average metrics for each atom count
            for atom_count, losses in metrics['per_atom_metrics'].items():
                if not losses['total_rl_rewards']:
                    continue
                avg_metrics = {k: np.mean(v) if v else 0.0 for k, v in losses.items()}
                epoch_metrics_json['atom_counts'][atom_count] = avg_metrics
                logging.info(
                    f"Epoch {epoch+1}, Atom Count {atom_count}: "
                    f"Avg Validity Score: {avg_metrics['validity_scores']:.4f}, "
                    f"Avg Fool Reward: {avg_metrics['fool_rewards']:.4f}, "
                    f"Avg Unique Mol Penalty: {avg_metrics['unique_mol_penalties']:.4f}, "
                    f"Avg Disconnect Penalty: {avg_metrics['disconnect_penalties']:.4f}, "
                    f"Avg Distribution Penalty: {avg_metrics['distribution_penalties']:.4f}, "
                    f"Avg Same Atom Penalty: {avg_metrics['same_atom_penalties']:.4f}, "
                    f"Avg Duplicate Penalty: {avg_metrics['duplicate_penalties']:.4f}, "
                    f"Avg Edge Density Penalty: {avg_metrics['edge_density_penalties']:.4f}, "
                    f"Avg Invalid Edge Penalty: {avg_metrics['invalid_edge_penalties']:.4f}, "
                    f"Avg Edge Count Penalty: {avg_metrics['edge_count_penalties']:.4f}, "
                    f"Avg Size Penalty: {avg_metrics['size_penalties']:.4f}, "
                    f"Avg Valence Penalty: {avg_metrics['valence_penalties']:.4f}, "
                    f"Avg Motif Reward: {avg_metrics['motif_rewards']:.4f}, "
                    f"Avg Similarity Reward: {avg_metrics['similarity_rewards']:.4f}, "
                    f"Avg Total RL Reward: {avg_metrics['total_rl_rewards']:.4f}"
                )

            # append epoch metrics to JSON data
            data_json.append(epoch_metrics_json)
            try:
                with open(per_atom_metrics_json_path, "w") as f:
                    json.dump(data_json, f, indent=4)
                logging.info(f"Epoch {epoch +1}: Saved per-atom metrics to {per_atom_metrics_json_path}")
            except Exception as e:
                logging.error(f"Epoch {epoch +1}: Failed to save per-atom metrics: {e}")
                traceback.print_exc()

            # clear per-atom metrics for the next epoch
            metrics['per_atom_metrics'].clear()

        # log average losses for pretraining
        logging.info(f"[Epoch {epoch +1}] (Pretraining) Avg D Loss: {avg_loss_D:.4f}, Avg G Loss: {-avg_loss_G:.4f}")
        
        # append generator and discriminator losses to loss file
        with open(gen_disc_loss_path, "a") as f:
            f.write(f"{epoch+1},avg_loss_D:{avg_loss_D},avg_loss_G:{avg_loss_G}\n")
        logging.info(f"[Epoch {epoch +1}] (Pretraining) Avg D Loss: {avg_loss_D:.4f}, Avg G Loss: {-avg_loss_G:.4f}")
