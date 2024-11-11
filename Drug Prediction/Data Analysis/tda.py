import os
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from Bio.PDB import PDBParser, MMCIFParser

# ignoring the warnings
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Suppress specific PDBConstructionWarnings about unrecognized records
warnings.simplefilter('ignore', PDBConstructionWarning)

# Function to calculate molecular descriptors for a molecule
def calculate_descriptors(molecule):
    descriptors = {
        'MolecularWeight': Descriptors.MolWt(molecule),
        'LogP': Descriptors.MolLogP(molecule),
        'HBDonors': Lipinski.NumHDonors(molecule),
        'HBAcceptors': Lipinski.NumHAcceptors(molecule),
        'RotatableBonds': Lipinski.NumRotatableBonds(molecule),
        'HeavyAtoms': Descriptors.HeavyAtomCount(molecule)
    }
    return descriptors

# Function to load a molecule from a PDB file
def load_molecule_from_pdb(pdb_path):
    mol = Chem.MolFromPDBFile(pdb_path, sanitize=True, removeHs=False)
    return mol if mol else None

# Function to get protein properties
def get_protein_properties(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure('Protein', pdb_path)
    num_residues = sum(1 for _ in structure.get_residues())
    num_atoms = sum(1 for _ in structure.get_atoms())
    return {'NumResidues': num_residues, 'NumAtoms': num_atoms}

# Processing directories and files
def process_directory(root_dir):
    all_data = []
    
	# the iterable for getting each of the pdb files
    iterable = list(os.walk(root_dir))
    
    for subdir, dirs, files in tqdm.tqdm(iterable, total=len(list(iterable))):
        for file in files:
            if file.endswith("_protein.pdb"):
                protein_path = os.path.join(subdir, file)
                protein_props = get_protein_properties(protein_path)
                protein_props['Type'] = 'Protein'
                all_data.append(protein_props)

            elif file.endswith("_ligand.pdb"):
                ligand_path = os.path.join(subdir, file)
                mol = load_molecule_from_pdb(ligand_path)
                if mol:
                    ligand_props = calculate_descriptors(mol)
                    ligand_props['Type'] = 'Ligand'
                    all_data.append(ligand_props)

            elif file.endswith("_pocket.pdb"):
                pocket_path = os.path.join(subdir, file)
                mol = load_molecule_from_pdb(pocket_path)
                if mol:
                    pocket_props = calculate_descriptors(mol)
                    pocket_props['Type'] = 'Pocket'
                    all_data.append(pocket_props)

    return pd.DataFrame(all_data)

# Function to plot histograms and save them to a directory
def plot_histograms(df, save_dir):
	os.makedirs(save_dir, exist_ok=True)  # Create the save directory if it doesn't exist
	
	for type_name, group in df.groupby('Type'):
		plt.figure(figsize=(15, 10))  # Slightly larger figure
		
		# Always use a grid that can fit 9 plots (3x3)
		for i, (column, data) in enumerate(group.drop(columns=['Type']).items(), 1):
			plt.subplot(3, 3, i)
			sns.histplot(data, kde=True, element='step', bins=30)
			plt.title(f'{type_name} {column}')
			plt.xlabel(column)
			plt.ylabel('Frequency')
		
		plt.tight_layout()
		save_path = os.path.join(save_dir, f'{type_name}_histogram.png')  # Generate the save path
		plt.savefig(save_path)  # Save the histogram as an image
		plt.close()  # Close the figure to free up memory

# Main script
if __name__ == "__main__":
	root_directory = '/Users/tristanbrigham/Work/Research/Gernstein/Alan/Alan_Gernstein_Pocket_Prediction/Pocket Prediction/refined-set'
	save_directory = '/Users/tristanbrigham/Work/Research/Gernstein/Alan/Alan_Gernstein_Pocket_Prediction/Drug Prediction/Data Analysis/GeneratedImages'
	df_descriptors = process_directory(root_directory)
	plot_histograms(df_descriptors, save_directory)




\begin{figure}[H]
    \centering
    \begin{subfigure}{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{ligand_bdacceptors.png}
        \caption{HB Acceptors Distribution}
        \label{fig:ligand_bdacceptors}
    \end{subfigure}
    \begin{subfigure}{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{ligand_hbdonors.png}
        \caption{HB Donors Distribution}
        \label{fig:ligand_hbdonors}
    \end{subfigure}
    \begin{subfigure}{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{ligand_heavyatoms.png}
        \caption{Heavy Atoms Distribution}
        \label{fig:ligand_heavyatoms}
    \end{subfigure}
    \caption{An example ligand for the 6PVE protein and its likely binding configuration in the protein}
\end{figure}

\vspace{-20}

\begin{figure}[H]
    \centering
    \begin{subfigure}{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{ligand_logp.png}
        \caption{LogP Distribution}
        \label{fig:ligand_logp}
    \end{subfigure}
    \begin{subfigure}{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{ligand_mol_weight.png}
        \caption{Molecular Weight Distribution}
        \label{fig:ligand_mol_weight}
    \end{subfigure}
    \begin{subfigure}{0.3\textwidth}
        \centering
        \includegraphics[width=\textwidth]{ligand_rotatablebonds.png}
        \caption{Rotatable Bonds Distribution}
        \label{fig:ligand_rotatablebonds}
    \end{subfigure}
    \caption{An example ligand for the 6PVE protein and its likely binding configuration in the protein}
\end{figure}