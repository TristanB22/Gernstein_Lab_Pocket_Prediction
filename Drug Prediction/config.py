# config.py

'''
The point of this file is to set up the configuration for the entire project
when the user actually runs the training of the model.
'''

import torch
import random
import numpy as np
import datetime
import os
import logging
import warnings

# the device that we are training on
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seed to make sure that the results that we get are reproducible
SEED = 42

# set the seed everywhere
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# basic model and training settings for things like logging
TEMPERATURE = 1.0
PRINT_ERROR = True
PRINT_INFO = True
TRAIN_ON_SMALL_DATASET = True
SMALL_DAT_NUM_MOLS = 10
VISUALIZE_MOLECULE_EVERY_EPOCH = True
VISUALIZE_EVERY_INSTANCE = False
WRITE_MOLECULE_IMAGES = True
LOG_RL_LOSSES = True

# hyperparameters for the models
NOISE_DIM = 128
HIDDEN_DIM = 512
NUM_LAYERS = 14
BATCH_SIZE = 128
EPOCHS = 700
LEARNING_RATE = 0.001
MAX_NODES = 15
RL_BATCH_SIZE = 256
REINFORCEMENT_LEARNING_FACTOR = 5
NUM_BASELINE_SAMPLES = 500
RL_SCALING_FACTOR = 10.0

# directories and paths to the most important files
GENERATOR_CHECKPOINT_PATH = './generator_latest_ckpt.pth'
PDBBIND_LIGAND_SDF_DIR = "/home/tjb73/project/Pocket Prediction/refined-set"
moses_csv_path = "./test.txt"

# how many molecules we should generate in the testing phase
NUM_MOLECULES_TO_GENERATE = 10000
SAVE_DIRECTORY = './GEN_EVAL_PDBbind'

# make the directory to make sure that it exists
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

# logging profile for this run of the program
LOGGING_ENABLED = True
CHECKPOINTING_ENABLED = True
SAVE_MODEL_EVERY_EPOCHS = 5
SUPER_LOGGING = False
PRETRAIN_EPOCHS = 2
ENABLE_REINFORCEMENT_LEARNING = True
LOAD_MODEL = False

# the reinforcement learning parameters that we should use for this run
FLAGS = {
    'ENABLE_VALIDITY_REWARD': False,
    'ENABLE_FOOL_REWARD': False,
    'ENABLE_UNIQUE_MOL_PENALTY': False,
    'ENABLE_MOTIF_REWARD': False,
    'ENABLE_SIMILARITY_REWARD': False,
    'ENABLE_VALENCE_CHECK': False,
    'ENABLE_VALENCE_PENALTY': False,
    'ENABLE_CONNECTIVITY_PENALTY': False,
    'ENABLE_DISTRIBUTION_PENALTY': False,
    'ENABLE_SAME_ATOM_PENALTY': False,
    'ENABLE_DUPLICATE_MOLECULE_PENALTY': False,
    'ENABLE_EDGE_DENSITY_PENALTY': False,
    'ENABLE_INVALID_EDGE_PENALTY': False,
    'ENABLE_EDGE_COUNT_PENALTY': False,
    'ENABLE_SIZE_PENALTY': False,
    'ENABLE_DISCONNECT_PENALTY': False,
}

# the scaling factors for the penalties and rewards in the reinforcement
# learning setup
VALIDITY_REWARD = 4.0
DISCONNECT_PENALTY = 0
DISTRIBUTION_PENALTY = 4.0
INVALID_EDGES_SCALING = 0.5
VALENCE_PENALTY_SCALING = 2.1
SAME_ATOM_PENALTY = 5.0
DUPLICATE_PENALTY = 1.5
EDGE_DENSITY_PENALTY = 10.0
LARGE_SIZE_THRESHOLD = 25
SIZE_SCALER = 0.5
VALENCE_PENALTY_SCALE = 0.05
UNIQUE_MOL_SCALER = 0.07
TRIPLE_RATIO_SCALING = 6.25
EDGE_COUNT_PENALTY_SCALING = 0.9
DIST_DISTRIBUTION_PENALTY = 0.05
FOOL_SCALING = 3.0

# suppressing warnings and making
# sure that we only get the output that is important and interesting
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")

# run_dir setup
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_dir = os.path.join('runs', f'run_{timestamp}')

# make the directories and get the paths
os.makedirs(run_dir, exist_ok=True)
os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)
loss_file_path = os.path.join(run_dir, "loss.txt")
gen_disc_loss_path = os.path.join(run_dir, "gen_loss.txt")
