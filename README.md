# Gernstein Lab Pocket Prediction

This project is a collaboration between myself (Tristan Brigham)[mailto:tristan.brigham@yale.edu] and (Alan Ianselli)[alan.ianeselli@yale.edu] in the Gernstein Lab at Yale University.

### Purpose of the Program

This project looks to use machine learning to predict binding pockets in proteins for ligand molecules to bond to. The purpose of the project is to decrease the time that it takes to find these pockets, and will be potentially expanded into programs looking at drug discovery and application. 


### Running this Program

In order to create and train the machine learning model, please run the `train_model.py` program. On the other hand, if you want to create the dataset (which is required before training the model), then please run the `creating_ligand_dataset.py` program with the required paths in place. 

### Overview of the Files

Here is a breakdown of all of the files that are in this repository and what they do. Please reference the chart to understand what files need to be changed to add functionality and get the entire process working. 

| File Name                   | Description                                                        |
|:---------------------------:|:------------------------------------------------------------------:|
| creating_ligand_dataset.py | This script pulls the data from the pdb file directories and creates the tensors that are going to be used to train the model. |
| train_model.py | This program is used to actually train the model using the data that is created with the creating_ligand_dataset.py program. |
