# this file is used to define the unet model
# so that we can initialize and train it in the root directory
# with the train_model.py file

import os
import pickle as pkl
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from torchviz import make_dot


# this loads in a tensor from a numpy file and converts it to a pytorch tensor
# path: the path to the file that has the tensor that we are loading
def load_numpy(path):
    return torch.tensor(np.load(path).astype(np.float32 ))

def standard_scaler(non_zero_indices, tensor):

	# standardize only the non-zero elements
	mean = torch.mean(tensor[non_zero_indices])
	std = torch.std(tensor[non_zero_indices])

	# add a small constant for numerical stability
	tensor[non_zero_indices] = (tensor[non_zero_indices] - mean) / (std + 1e-8)  

	return tensor

def min_max_scaler(non_zero_indices, tensor):
	
	# max and min vals
	max_ = torch.max(tensor[non_zero_indices])
	min_ = torch.min(tensor[non_zero_indices])

	# scale only the non-zero elements
	# add a small constant to avoid division by zero
	tensor[non_zero_indices] = (tensor[non_zero_indices] - min_) / (max_ - min_ + 1e-8)  

	return tensor

def   mean_max_scaler(non_zero_indices, tensor):
    
	# max and min values
	max_ = torch.max(tensor[non_zero_indices])
	min_ = torch.min(tensor[non_zero_indices])
	mean_ = torch.mean(tensor[non_zero_indices])
     
	# scale and shift only the non-zero elements
	# add a small constant for stability
	tensor[non_zero_indices] = (tensor[non_zero_indices] - mean_) / (max_ - min_ + 1e-8)  
     
	return tensor

def max_scaler(non_zero_indices, tensor):
    
	# get max value
	max_ = torch.max(tensor[non_zero_indices])

	# normalize only non-zero elements to their maximum value
	# prevent division by zero
	tensor[non_zero_indices] = tensor[non_zero_indices] / (max_ + 1e-8)  

	return tensor

# the custom metric that we are using
def UNet_evaluating_custom_metric(y_true, y_pred, eta = 1e-11):
    
	# get the binary mask
    mask = y_true > eta
    
	# return the mean and abs of diff
    return torch.mean(torch.abs(y_true[mask] - y_pred[mask]))




# this function loads in all of the data that is associated with a molecule
# it is optimized for the old versions of the files that used numpy tensors
# given the path and returns the features + labels
# path: the path to the molecule that we are trying to load in
# parent_dir: the parent directory where the directory of molecules is
def numpy_open_files_select_molecule_path(path, parent_dir=False):
	
	# non-zero value to replace zeros with
	nzv = 0.0  

	# update path to be relative
	# path = os.path.join("..", path)

	# open the features of the observation
	# ensure it is a float tensor
	N = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/N_tensor.npy")).astype(np.float32))

	# indices of non-zero values (i.e., protein)
	non_zeros = torch.nonzero(N, as_tuple=True)  
	
	# indices of zero values (i.e., outside)
	zeros = torch.nonzero(N == 0, as_tuple=True) 

	# set protein to 1
	N[non_zeros] = 1.0  
	
	# set outside to nzv
	N[zeros] = nzv  

	# add an extra dimension for the later merging, to create the batch
	N = N.unsqueeze(-1)  

	bfactors = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/bfactors_tensor.npy")).astype(np.float32))
	# bfactors = bfactors #make sure it is a float tensor
	bfactors = mean_max_scaler(non_zeros, bfactors) #normalize between 0 and 1 (+ outliers)
	bfactors[zeros] = nzv #outside gets nzv
	bfactors = bfactors.unsqueeze(-1)

	#repeat for all the other features
	buriedness = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/buriedness_tensor.npy")).astype(np.float32))
	# buriedness = buriedness 
	buriedness = mean_max_scaler(non_zeros, buriedness)
	buriedness[zeros] = nzv
	buriedness = buriedness.unsqueeze(-1)

	charge = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/charge_tensor.npy")).astype(np.float32))
	# charge = charge
	charge = mean_max_scaler(non_zeros, charge)
	charge[zeros] = nzv
	charge = charge.unsqueeze(-1)

	radius = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/radius_tensor.npy")).astype(np.float32))
	# radius = radius 
	radius = mean_max_scaler(non_zeros, radius)
	radius[zeros] = nzv
	radius = radius.unsqueeze(-1)

	hbdon = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/hbdon_tensor.npy")).astype(np.float32))
	# hbdon = hbdon 
	hbdon = mean_max_scaler(non_zeros, hbdon)
	hbdon[zeros] = nzv
	hbdon = hbdon.unsqueeze(-1)

	hbacc = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/hbac_tensor.npy")).astype(np.float32))
	# hbacc = hbacc 
	hbacc = mean_max_scaler(non_zeros, hbacc)
	hbacc[zeros] = nzv
	hbacc = hbacc.unsqueeze(-1)

	sasa = torch.from_numpy(np.load(os.path.join(path, "for_labview_protein_/sasa_tensor.npy")).astype(np.float32))
	# sasa = sasa 
	sasa = mean_max_scaler(non_zeros, sasa) #NOTE: could consider recalculating the non_zero indices just for sasa, as the inside of the protein contains zeros
	sasa[zeros]= nzv
	sasa = sasa.unsqueeze(-1)

	# get all of the tensors together now
	features = torch.cat([N, bfactors, buriedness, charge, radius, hbdon, hbacc, sasa], dim=-1)

	# target feature
	target=torch.from_numpy(np.load(os.path.join(path, "for_labview_pocket_/N_tensor_new_pocket.npy")))
	non_zeros_target = (torch.nonzero(target, as_tuple=True)) #calculate the non_zero values of the target (i.e. pocket)

	eta=0.00000000001
	target[non_zeros]+= eta  #to distinguish it from the zeros of the outside (for metrics calculation)
	
	# make it binary, the pocket becomes 1
	target[non_zeros_target] = 1.0 
      
	# everything that is outside in the observation becomes nvz, also to mask out the wrong hydrogens added to the pocket
	target[zeros] = nzv 

	# add the batch dimension to the outside
	target = target.unsqueeze(-1)

	return features, target



# this function loads in all of the data that is associated with a molecule
# and requires that the fixing_numpy_data.py code is called before running
# this yields a 5x speedup over the function above for loading the tensors in 
# given the path and returns the features + labels
# path: the path to the molecule that we are trying to load in
# parent_dir: the parent directory where the directory of molecules is
def torch_open_files_select_molecule_path(path, parent_dir=False):
	
	features = torch.load(os.path.join(path, f"{path.replace('../refined-set/', '')}_total_features.pt"))
	target = torch.load(os.path.join(path, f"{path.replace('../refined-set/', '')}_total_target.pt"))

	return features, target



# create a dataset class for the pockets and the molecules that we are predicting the pockets for
# please make sure to set the number of workers as higher in this dataset to ensure that 
# we are not constrained on the IO operations side of things
class PocketDataset(Dataset):
    
	# the file paths to all of the files 
    def __init__(self, filepaths):
        self.filepaths = filepaths

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx].strip()
        features, target = numpy_open_files_select_molecule_path(path)
        return features, target




# define a class that allows for us to include channel-wise attention in the tensor that
# we are feeding to the model 
class ChannelAttention(nn.Module):
    
	# initialize and run the attention for the channels that we are inputting to the model
    def __init__(self, channels):
        
		# initialize the parent
        super(ChannelAttention, self).__init__()
        
		# define the gamma for the attention computation
        self.gamma = nn.Parameter(torch.ones(1))


	# define the forward pass for the attention
    def forward(self, inputs):
        
		# get each of the components of the input
		# batch size (b)
		# depth (l)
		# height (h)
		# width (w)
		# channels (c)
        b, l, h, w, c = inputs.shape
        
        # reshape
        x = inputs.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(b, c, l*h*w)

        # get the softmax of the query and key cross product
        x_mutmul = torch.matmul(x.transpose(1, 2), x)
        x_mutmul = F.softmax(x_mutmul, dim=-1)

        # apply the attention with the value dot prod
        x = torch.matmul(x, x_mutmul)
        x = x.view(b, c, l, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()

        # scale and keep the residual
        x = x * self.gamma
        x = x + inputs
        return x


# define a class that incorporates position attention into the model
# through this layer
class PositionAttention(nn.Module):
    
	# initialize the positional autoencoder
    def __init__(self, channels):
        
		# init and call the parent class
        super(PositionAttention, self).__init__()
        
		# define the layers that we are using
        self.conv1 = nn.Conv3d(channels, channels // 2, kernel_size=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))

	# define the forward pass for the attention layer
    def forward(self, inputs):
        
		# get the inputs to the layer
		# batch size (b)
		# depth (l)
		# height (h)
		# width (w)
		# channels (c)
        b, l, h, w, c = inputs.shape

		# convolve over the inputs while reshaping the output
		# so that the channel layer is the first which allows for slightly
		# faster tensor operations
        x1 = self.conv1(inputs).permute(0, 4, 1, 2, 3).contiguous()
        x1 = x1.view(b, c//2, l*h*w)
        
		# computing the self attention of the input
        x1_mutmul = torch.matmul(x1.transpose(1, 2), x1)
        x1_mutmul = F.softmax(x1_mutmul, dim=-1)

		# transform the shape of the output tensor again
        x2 = self.conv2(inputs).permute(0, 4, 1, 2, 3).contiguous()
        x2 = x2.view(b, c, l*h*w)

		# compute the self attention without reducing the feature space
        x2_mutmul = torch.matmul(x2, x1_mutmul.transpose(1, 2))
        x2_mutmul = x2_mutmul.view(b, c, l, h, w)
        x2_mutmul = x2_mutmul.permute(0, 2, 3, 4, 1).contiguous()

		# add the residual connections to the network
        x2_mutmul = x2_mutmul * self.gamma
        x = x2_mutmul + inputs
        
		# return the final output
        return x


# compute multi-head attention with the channel and the positional information
class DANet(nn.Module):
    
	# initialize the two-part attention
	def __init__(self, channels):
		
		super(DANet, self).__init__()
		
		# init the attention computation
		self.channel_attention = ChannelAttention(channels)
		self.position_attention = PositionAttention(channels)

	# compute the forward pass of the attention
	def forward(self, inputs):
            
		# get the two attentions and add them together
		x1 = self.channel_attention(inputs)
		x2 = self.position_attention(inputs)
            
		# sum
		x = x1 + x2
            
		return x






# defining the UNet architecture below
# it is going to be made from a series of up-sampling and down-sampling 3D blocks with residual connections
# this version of the model is not going to include attention blocks
class UNet(nn.Module):

	# current defined architecture
	'''
	UNET full

	downsampling
	[32,32,32,8] #input, skip connection B
	[16,16,16,32] #skip connection A

	[8,8,8,64] #bottom

	upsampling
	[16,16,16,32] #skip connection A
	[32,32,32,8] #skip connection B

	[32,32,32, 1]
	'''
      
	# initializing the network
	# use_attn_2: whether we should use attention in this network or not
	# dropout_rate: the rate at which we do dropout during training for the network
	def __init__(self, use_attn_2=False, dropout_rate=0.2):
		
		# initialize the model class
		super(UNet, self).__init__()
		
		# define the unet architecture
		self.input_channels = 32
            
		# keep track of whether we should use attention:
		self.use_attn_2 = use_attn_2
            
		# start with the downsampling of the information while keeping track of the residuals for skip connections
		self.encoder_1 = nn.Sequential(
			nn.Conv3d(self.input_channels, out_channels=32, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm3d(32),
			nn.LeakyReLU(),
			nn.Dropout(dropout_rate),
		)
		
		# define a second encoder to compress the input information even further
		self.encoder_2 = nn.Sequential(
			nn.Conv3d(in_channels=32, out_channels=128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm3d(128),
			nn.LeakyReLU(),
			nn.Dropout(dropout_rate),
		)
			
		# if we are including attention in the network then include it here
		if self.use_attn_2:
			
			# use the DaNet attention for the positional with channel
			self.danet_attn_layer_2 = DANet(channels=128)

		# up-sampling portion of the model
		# this one corresponds with the encoder_2 that we defined above
		self.decoder_2 = nn.Sequential(
			nn.ConvTranspose3d(in_channels=128, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
		)
		
		self.decoder_1 = nn.Sequential(
			nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
		)

		# combining the residual information from the input and the computed output
		self.combine_conv = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate)
        )
            
		# output head with final information
		self.output_conv = nn.Sequential(
			nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0), # this is meant to reduce the last dimension
			nn.Sigmoid()
		)


	# the forward pass of the master model
	def forward(self, x):
            
		# encoding the input into a latent space
		d1 = self.encoder_1(x)
		d2 = self.encoder_2(d1)

		# apply attention if that is something that we are using in this model
		if self.use_attn_2:
			d2 = self.danet_attn_layer_2(d2)
		
		# upsample all of the information
		u1 = self.decoder_2(d2)
		
		# concat the residual connections for more information
		u1 = torch.cat((d1, u1), dim=1)
		
		# upsample the input
		out = self.decoder_1(u1)
            
		# concat again with the first downsample
		out = torch.cat((x, out), dim=1)
		
		# combine the residual and the upsampled information
		out = self.combine_conv(out)
			
		# return the output with the permuted order to bring the last dimension to the front
		out = out.permute(0, 4, 1, 2, 3)  # new shape: [16, 8, 32, 32, 32]

		# apply the convolution layer and sigmoid
		out = self.output_conv(out)

		# permute back to the original order [batch, depth, height, width, channels]
		out = out.permute(0, 2, 3, 4, 1)
		
		return out
      
	
	# this function shows the structure of the model that we have defined to the user
	# so that we can understand the skip connections and structure of the overall model
	def show_model_structure(self):
		
		# random input to get the structure of the model
		dummy_input = torch.randn(1, self.input_channels, 32, 32, 32)  # Adjust the size according to the expected input
		
		# forward pass through the model to get the output
		model_output = self(dummy_input)
		
		# get the output graph and show it to the user
		model_graph = make_dot(model_output, params=dict(list(self.named_parameters())))
		model_graph.view()




# get the coordinates for the predicted tensor so that we can save 
# the output to a file and save the prediction
def obtain_coordinates(pdb, predicted_tensor):
    
    # move the tensor from the compute module
    predicted_tensor = predicted_tensor[0, :, :, :, 0].cpu()

    # load in the bin boundaries that we are using
    boundaries = np.loadtxt(f"../refined-set/{pdb}/for_labview_protein_/axis_bins.txt")
    x_bins, y_bins, z_bins = boundaries

    # load in the protein and pocket coordinates
    xyz_protein = np.loadtxt(f"../refined-set/{pdb}/for_labview_protein_/xyz.txt")
    xyz_pocket_target = np.loadtxt(f"../refined-set/{pdb}/for_labview_pocket_/xyz_new_pocket.txt")

    # find the indicies in the bins
    x_indices = torch.bucketize(torch.tensor(xyz_protein[:, 0]), torch.tensor(x_bins))
    y_indices = torch.bucketize(torch.tensor(xyz_protein[:, 1]), torch.tensor(y_bins))
    z_indices = torch.bucketize(torch.tensor(xyz_protein[:, 2]), torch.tensor(z_bins))

    # change the indexes so that they are zero indexed
    x_indices = x_indices - 1
    y_indices = y_indices - 1
    z_indices = z_indices - 1

    # get the predicted values
    predicted_values = predicted_tensor[x_indices, y_indices, z_indices]

	# return the relevant information
    return xyz_protein, predicted_values.numpy(), xyz_pocket_target



# this function helps us understand what the generated protein actually looks like
# against the predicted values with the pocket
def visualize_protein(xyz_protein, predicted_values, xyz_pocket_target, pdb):
    
    # convert the input if they are tensors or something else
    if isinstance(xyz_protein, torch.Tensor):
        xyz_protein = xyz_protein.cpu().detach().numpy()
    if isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.cpu().detach().numpy()
    if isinstance(xyz_pocket_target, torch.Tensor):
        xyz_pocket_target = xyz_pocket_target.cpu().detach().numpy()

    # plot the predicted proteins
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(xyz_protein[:, 0], xyz_protein[:, 1], xyz_protein[:, 2], c=predicted_values, marker='.', alpha=0.8, cmap=plt.cm.copper, s=np.array(predicted_values) * 20 + 0.05, vmin=0, vmax=1)
    ax.axis('off')
    plt.title("Prediction " + pdb)
    fig.colorbar(p, ax=ax)

    # now plot the target 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz_pocket_target[:, 0], xyz_pocket_target[:, 1], xyz_pocket_target[:, 2], marker='.', alpha=0.8, c="sandybrown", s=20)
    ax.scatter(xyz_protein[:, 0], xyz_protein[:, 1], xyz_protein[:, 2], marker='.', alpha=0.3, c="k", s=1)
    ax.axis('off')
    plt.title("Target " + pdb)
    plt.show()
    