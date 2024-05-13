# Gernstein Lab Pocket Prediction

This project is a collaboration between myself [Tristan Brigham](mailto:tristan.brigham@yale.edu) and [Alan Ianseelli](mailto:alan.ianeselli@yale.edu) in the Gernstein Lab at Yale University.


### Purpose of the Program

This project looks to use machine learning to predict binding pockets in proteins for ligand molecules to bond to. The purpose of the project is to decrease the time that it takes to find these pockets, and will be potentially expanded into programs looking at drug discovery and application. 


### Running this Program

Before running the program, please make sure to download all of the packages in the `requirements.txt file`.

In order to create and train the machine learning model, please run the `train_model.py` program. On the other hand, if you want to create the dataset (which is required before training the model), then please run the `creating_ligand_dataset.py` program with the required paths in place. 

The model training will use the molecules pointed to by the paths in `training_molecules_filepaths.txt`. These files are too numerous and large to include in this repository, but can be downloaded from PDB. 

An additional note for Mac users looking to use the MPS speedup: as of the release of this repository, the torch official repository does not support 3D convolutions on the MPS platform. This program uses an installation of torch with [this pull request](https://github.com/pytorch/pytorch/pull/99246) in order to allow the 3D convolutions required for the successful execution of this program to happen.

### Evaluating Models
Once you have trained the model and have a saved version, you can evaluate any saved model by running `evaluate_model.py` with the right path specified in the global variables of the file. 

### Overview of the Files

Here is a breakdown of all of the files that are in this repository and what they do. Please reference the chart to understand what files need to be changed to add functionality and get the entire process working. 

| File Name                   | Description                                                        |
|:---------------------------:|:------------------------------------------------------------------:|
| `creating_ligand_dataset.py` | This script pulls the data from the pdb file directories and creates the tensors that are going to be used to train the model. Running this script on the original pdb dataset that was created with the PyMol scripts yields a 4x speedup in the training and processing time of all of the input information. |
| `train_model.py` | This program is used to actually train the model using the data that is created with the creating_ligand_dataset.py program. |
| `evaluate_model.py` | This program runs the model that is pointed to by the file path in the program and evaluates the performance on a benchmark molecule that we define. |
| `requirements.txt` | As is standard for python repositories, this contains all of the packages that are required to run the entire program. Please run `pip install -r requirements.txt` in order to install all of the packages found in this file. |
| `run_attn_model.sh` | This is the script that is submit to the Yale HPC system in order to run the machine learning model training script. Although it is optimized for Yale's high performance computing cluster, it can be modified to fit other systems. |
| `training_molecules_filepaths.txt` | This file contains all of the protein-ligand combinations that are going to be used to train the machine learning model that we are building. |
| `UNet_Attention_Model` | This directory contains all of the definitions for the model. It can be imported as a package and trained with the `train_model.py` python program. |
| `Trained_Models` | This directory contains all of the models that we have finished training and saved. They can be loaded in using the normal loading functions found in the UNet Model definitions. |
| `Benchmarking` | This directory contains the code that I use to benchmark new against old methods for loading data and training the model. It is a proving ground for the code and is rather unstructured. Please forgive the messiness of the code as it was coded rather spontaneously. |
| `PyMol_Scripts` | This directory contain the scripts that are used to interface with PyMol in order to create the tensors that are being processed by the model we are training. They are called by other scripts in previous iterations of the program and come from Alan. |
