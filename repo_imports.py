# creating an imports file so that we can keep track of all of the packages that
# are imported in the program for safety and security

# common imports
import os
import sys

# data manipulation and visualization imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# protein package imports
from mdtraj import load
from rdkit import Chem
from rdkit.Chem import AllChem



# standard functions that are helpful
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
