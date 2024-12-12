# utils/plotting.py

'''
Plotting utilities for visualizing molecular data.

This module provides functions to plot the distribution of atom types and to visualize
generated molecules. It utilizes matplotlib for creating bar charts and RDKit for rendering
molecular structures.
'''

import matplotlib.pyplot as plt
import logging
import os
from rdkit.Chem import Draw
import traceback
from config import PRINT_INFO, run_dir
from rdkit import Chem


def plot_atom_type_distribution(distribution_counter, title="Atom Type Distribution"):
    '''
    Plots the distribution of atom types in the dataset.
    
    :param distribution_counter [dict]: A dictionary with atom types as keys and their counts as values.
    :param title [str, optional]: The title of the plot. Defaults to "Atom Type Distribution".
    :return [None]: Saves the plot as a PNG image in the designated directory.
    '''
    try:
        # check if distribution data is available
        if not distribution_counter:
            logging.warning("no atom type distribution data available for plotting.")
            return

        # extract atom types and their corresponding counts
        atom_types = list(distribution_counter.keys())
        counts = list(distribution_counter.values())

        # create a bar plot for atom type distribution
        plt.figure(figsize=(8, 6))
        plt.bar(atom_types, counts, color='skyblue')
        plt.xlabel('Atom Types')
        plt.ylabel('Counts')
        plt.title(title)
        plt.tight_layout()

        # define the path to save the plot
        plot_path = os.path.join(run_dir, "images", f"atom_distribution_{title.replace(' ', '_')}.png")
        plt.savefig(plot_path)
        plt.close()

        # log the successful saving of the plot
        logging.info(f"atom type distribution plot saved to {plot_path}.")

        # optionally print information to the console
        if PRINT_INFO:
            print(f"saved atom type distribution plot to {plot_path}.")

    except Exception as e:
        # log any errors encountered during plotting
        logging.error(f"error plotting atom type distribution: {e}")
        traceback.print_exc()


def visualize_molecule(mol, epoch, run_dir, title="Generated Molecule"):
    '''
    Visualizes and saves an image of the generated molecule.
    
    :param mol [rdkit.Chem.Mol]: The molecule to visualize.
    :param epoch [int]: The current training epoch number.
    :param run_dir [str]: The directory where the image will be saved.
    :param title [str, optional]: The title of the molecule image. Defaults to "Generated Molecule".
    :return [None]: Saves the molecule image as a PNG file in the designated directory.
    '''
    # check if the molecule is valid
    if mol is None:
        logging.warning(f"cannot visualize molecule: {title} is none.")
        return
    try:
        # convert the molecule to a SMILES string for logging
        smiles = Chem.MolToSmiles(mol)
        logging.info(f"[visualize_molecule] smiles: {smiles}")
        
        # generate an image of the molecule
        img = Draw.MolToImage(mol, size=(300, 300))
        plt.figure(figsize=(4, 4))
        plt.imshow(img)
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        
        # attempt to display the plot
        try:
            plt.show()
        except:
            pass

        # define the filename and path for saving the image
        image_filename = f"generated_molecule_epoch_{epoch + 1}.png"
        image_path = os.path.join(run_dir, "images", image_filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        
        # save the molecule image
        img.save(image_path)
        plt.close()
        
        # log the successful saving of the molecule image
        logging.info(f"[visualize_molecule] saved image as {image_path}")
    except Exception as e:
        # log any errors encountered during visualization
        logging.error(f"[visualize_molecule] failed to visualize molecule: {e}")
        traceback.print_exc()
