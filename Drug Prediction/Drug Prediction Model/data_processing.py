# data_processing.py

import os
import torch
import logging
from tqdm import tqdm
from torch_geometric.data import Dataset
from torch_geometric.utils import from_networkx
from graph_generation import MolGraphGeneration, MolEncoder
from rdkit import Chem


class MoleculeDataset(Dataset):
    
    def __init__(self, root, transform=None, pre_transform=None):

        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  

        # print the log level from env
        print(f"Log level in env: {log_level}")
        print(f"Assigned log level: {log_level}")

        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [DATASET] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.logger = logging.getLogger(__name__)

        # log that we are starting the dataset loaders
        self.logger.info("Starting dataset loaders...")
        
        self.directory = root
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        self.data_list = []
        self.total_molecules = 0
        self.successful_molecules = 0

        self.process()


    @property
    def raw_file_names(self):

        """
        Generate a list of all the SDF files in the dataset directory.

        :ret [list[str]]: List of all SDF files in the dataset directory
        """

        sdf_files = []
        for root, _, files in os.walk(self.directory):
            sdf_files.extend([os.path.join(root, f) for f in files if f.endswith('.sdf')])
        return sdf_files


    @property
    def processed_file_names(self):
        
        """
        Generate a list of all the processed files in the dataset directory.

        :ret [list[str]]: List of all processed files in the dataset directory
        """

        return [f'{os.path.splitext(os.path.basename(file))[0]}_processed.pt' for file in self.raw_file_names]


    def process(self):

        """
        Function to process the dataset and generate objects for ecah molecule in the dataset.

        :ret [None]
        """

        # the list of the molecules that we are going to use
        self.data_list = []
        
        # the total number of molecules that we process
        self.total_molecules = 0

        # the number of molecules that we successfully processed
        self.successful_molecules = 0

        # check if we are going to use tqdm or not
        use_tqdm = self.logger.level <= logging.ERROR

        # get the testing environment variable
        testing_env = os.getenv('TESTING', 'false').lower()

        # check if the testing environment variable is valid
        if testing_env not in ['true', 'false']:
            raise ValueError("Invalid value for TESTING environment variable. Must be 'true' or 'false'.")

        # iterate through the files and process them
        # with the tqdm if we are not in the testing environment
        if testing_env == 'true':
            file_iterator = tqdm(self.raw_file_names[:100], desc="Processing SDF files") if use_tqdm else self.raw_file_names[:100]
        else:
            file_iterator = tqdm(self.raw_file_names, desc="Processing SDF files") if use_tqdm else self.raw_file_names

        # iterate through the files and process them
        for sdf_file in file_iterator:

            # load the molecules from the sdf file
            supplier = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)

            # iterate through the molecules and process them
            for mol in supplier:
                
                # keep track of the number of molecules that we process
                self.total_molecules += 1  
                
                # make sure that there is a molecule there
                if mol is None:
                    self.logger.warning(f"Skipping invalid molecule in file: {sdf_file}")
                    continue 

                # sanitize the molecule
                try:
                    Chem.SanitizeMol(mol)

                except (Chem.KekulizeException, Chem.AtomValenceException) as e:
                    self.logger.warning(f"Skipping unsanitizable molecule in file: {sdf_file}. Error: {e}")
                    continue  
                
                # get the backbone of the molecule
                backbone_graph = MolGraphGeneration.get_backbone_graph(mol)
                if not backbone_graph or len(backbone_graph.nodes) == 0:
                    self.logger.warning(f"Skipping empty backbone graph for molecule in file: {sdf_file}")
                    continue  
                
                # make it a ptgeo object dataset
                data = from_networkx(backbone_graph)
                
                # make sure that there are features
                if len(backbone_graph.nodes) > 0:

                    # stack the features for the input to the model
                    data.x = torch.stack([attr['features'] for _, attr in backbone_graph.nodes(data=True)])

                else:
                    self.logger.warning(f"Skipping molecule with empty node features in file: {sdf_file}")
                    continue

                # check for edges
                if len(backbone_graph.edges) > 0:
                    data.edge_attr = torch.stack([attr['features'] for _, _, attr in backbone_graph.edges(data=True)])
                    data.edge_index = torch.tensor(list(backbone_graph.edges)).t().contiguous()

                else:
                    # get the default edge attributes
                    data.edge_attr = torch.zeros((0, MolEncoder.bond_features(next(mol.GetBonds())).shape[0]))
                    data.edge_index = torch.empty((2, 0), dtype=torch.long)
                
                # add the data to the list
                data.y = torch.tensor([1])  
                self.data_list.append(data)
                self.successful_molecules += 1  

        # check if we have any molecules and if so how many we have
        success_percentage = (self.successful_molecules / self.total_molecules) * 100 if self.total_molecules > 0 else 0
        self.logger.log(logging.INFO, f"Processed {self.successful_molecules} out of {self.total_molecules} molecules successfully.")
        print(f"Processed {self.successful_molecules} out of {self.total_molecules} molecules successfully.")
        self.logger.log(logging.INFO, f"Success rate: {success_percentage:.2f}%")
        print(f"Success rate: {success_percentage:.2f}%")

    # return the length of the dataset
    def len(self):
        return len(self.data_list)

    # get the data at a specific index
    def get(self, idx):
        return self.data_list[idx]
