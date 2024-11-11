# train_gan.py

import os
import logging
from dotenv import load_dotenv
from data_processing import MoleculeDataset
from generator import Generator
from discriminator import Discriminator
from visualization import visualize_generated_molecule, visualize_dataset_sample

from rdkit import Chem
from rdkit import RDLogger

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm

# configure logging using the environment variables
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format='%(asctime)s [TRAINING] %(levelname)s: %(message)s',
    level=log_level
)
logger = logging.getLogger("GAN_Trainer")

# configure RDKit logger to suppress warnings
# based on the log level
rdkit_logger = RDLogger.logger()
if log_level in ["ERROR", "CRITICAL"]:
	print(f"Setting RDKIT logger to: CRITICAL")
	rdkit_logger.setLevel(RDLogger.CRITICAL)
else:
	print(f"Setting RDKIT logger to: INFO")
	rdkit_logger.setLevel(RDLogger.INFO)

# hyperparameters
NOISE_DIM = 128
HIDDEN_DIM = 256
NUM_NODE_FEATURES = 7  
NUM_EDGE_FEATURES = 1  
NUM_LAYERS = 3
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0002
MAX_NODES = 50  # the max number of atoms that we could have in the backbone of the molecule

# load environment variables
load_dotenv()

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# get the data path from the environment
data_path = os.getenv("REFERENCE_MOLECULES_PATH")
logger.info(f"Data path: {data_path}")

# initialize dataset and dataloader
logger.info("Initializing dataset and dataloader...")
dataset = MoleculeDataset(root=data_path)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# initialize models
logger.info("Initializing generator and discriminator models...")
generator = Generator(
    noise_dim=NOISE_DIM,
    hidden_dim=HIDDEN_DIM,
    num_node_features=NUM_NODE_FEATURES,
    num_edge_features=NUM_EDGE_FEATURES,
    max_nodes=MAX_NODES,
    num_layers=NUM_LAYERS
).to(device)

discriminator = Discriminator(
    input_dim=NUM_NODE_FEATURES,  
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
).to(device)

# loss and opt
criterion = nn.BCELoss()
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE)

# get how often we should visualize
visualize_every = int(os.getenv("VISUALIZE_EVERY", 10))

# print how often we are going to visualize
logger.info(f"Visualizing every {visualize_every} epochs")

# run the training of the models
logger.info("Starting training loop...")
for epoch in range(EPOCHS):

	# run through all of the data
	for real_data in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
		
		# set things up with the device and length informaiton
		real_data = real_data.to(device)
		batch_size = real_data.num_graphs

		# real and fake labels
		real_labels = torch.ones(batch_size, 1).to(device)
		fake_labels = torch.zeros(batch_size, 1).to(device)

		# train the disc
		discriminator.zero_grad()

		# run the real data
		outputs_real = discriminator(real_data.x, real_data.edge_index, real_data.batch)
		loss_real = criterion(outputs_real, real_labels)

		# fake data
		generated_data_list = generator(batch_size)
		fake_batch = Batch.from_data_list(generated_data_list).to(device)
		outputs_fake = discriminator(fake_batch.x, fake_batch.edge_index, fake_batch.batch)
		loss_fake = criterion(outputs_fake, fake_labels)

		# get the total loss with backprop
		loss_D = loss_real + loss_fake
		loss_D.backward()
		optimizer_D.step()

		# train the generator
		generator.zero_grad()

		# get the fake data
		generated_data_list = generator(batch_size)
		fake_batch = Batch.from_data_list(generated_data_list).to(device)
		outputs = discriminator(fake_batch.x, fake_batch.edge_index, fake_batch.batch)

		# generator fools the discriminator
		loss_G = criterion(outputs, real_labels)
		loss_G.backward()
		optimizer_G.step()

	logger.info(f"Epoch [{epoch+1}/{EPOCHS}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

	# visualize the generated molecules
	if (epoch + 1) % visualize_every == 0:
		logger.info("Visualizing generated molecule...")
		visualize_generated_molecule(generator, num_nodes=MAX_NODES)
