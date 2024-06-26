# Gernstein Lab Pocket Prediction

This project is a collaboration between myself [Tristan Brigham](mailto:tristan.brigham@yale.edu) and [Alan Ianeselli](mailto:alan.ianeselli@yale.edu) in the Gerstein Lab at Yale University.


### Purpose of the Program

This project looks to use machine learning to predict candidate drugs for a given ligand/pocket for the improved synthesis and implementation of drugs for medical care. 


### Running this Program

Before running the program, please make sure to download all of the packages in the `requirements.txt` file.


### Evaluating Models
Once you have trained the model and have a saved version, you can evaluate any saved model by running `evaluate_model.py` with the right path specified in the global variables of the file. 

### Overview of the Files

Here is a breakdown of all of the files that are in this repository and what they do. Please reference the chart to understand what files need to be changed to add functionality and get the entire process working. 

| File Name                   | Description                                                        |
|:---------------------------:|:------------------------------------------------------------------:|
| `train_model.py` | This program is used to actually train the model using the data that is created with the creating_ligand_dataset.py program. |
| `evaluate_model.py` | This program runs the model that is pointed to by the file path in the program and evaluates the performance on a benchmark molecule that we define. |
| `requirements.txt` | As is standard for python repositories, this contains all of the packages that are required to run the entire program. Please run `pip install -r requirements.txt` in order to install all of the packages found in this file. |
| `run_attn_model.sh` | This is the script that is submit to the Yale HPC system in order to run the machine learning model training script. Although it is optimized for Yale's high performance computing cluster, it can be modified to fit other systems. |
| `training_molecules_filepaths.txt` | This file contains all of the protein-ligand combinations that are going to be used to train the machine learning model that we are building. |
| `testing_molecules_filepaths.txt` | This file contains the paths to the protein-ligand combinationns that will be used to evaluate the effectiveness of the model that we have built. |
