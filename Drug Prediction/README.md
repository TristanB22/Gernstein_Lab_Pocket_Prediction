# Gernstein Lab Pocket Prediction

This project is a collaboration between myself [Tristan Brigham](mailto:tristan.brigham@yale.edu) and [Alan Ianeselli](mailto:alan.ianeselli@yale.edu) in the Gerstein Lab at Yale University.


### Purpose of the Program

This project looks to use machine learning to predict candidate drugs for a given ligand/pocket for the improved synthesis and implementation of drugs for medical care. 


### Running this Program

Before running the program, please make sure to download all of the packages in the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

You then have to get the dataset that you are going to work with from the appropriate website. For example, the [PDBbind Dataset](http://www.pdbbind.org.cn/) and the (Moses Dataset)[https://github.com/molecularsets/moses] can be found online. Once you have downloaded the appropriate dataset, follow the steps below:

After downloading the dataset:

- For PDBbind, navigate to the `Pocket Prediciton/PyMol_Scripts` directory and run the data preprocessing scripts. Beware running these locally as they can take over 20 hours to run with larger dataset. 
- For MOSES, place the downloaded SMILES file in the appropriate data_files/ directory and run the processing script to convert the SMILES into a usable format.

Once preprocessing is complete, adjust config.py to point to the correct dataset directories, then run:

```
python train_model.py --dataset [pdbbind or moses]
```

If you want to change any of the hyperparameters for the model, please change them in `config.py` or `main.py`.


### Directory Structure
```
home/
│
├── main.py                 # Entry point to run the entire project
├── config.py               # Central file for hyperparameters and settings
├── README.md               # This documentation
├── requirements.txt        # Python dependencies
├── models/
│   ├── __init__.py
│   ├── generator.py        # Generator model implementation
│   ├── discriminator.py    # Discriminator model implementation
│   └── utils.py            # Residual layers, Baseline class
├── data/
│   ├── __init__.py
│   ├── constants.py        # Allowed atom types, valence dict, motif SMARTS
│   ├── process.py          # Functions for loading and processing data
│   └── utils.py            # MolEncoder, MolGraphGeneration, and utilities
├── rl/
│   ├── __init__.py
│   ├── rewards.py          # Functions for computing various rewards
│   ├── penalties.py        # Functions for computing penalties
│   └── utils.py            # Utility functions for RL (e.g., apply_flags_to_reward)
└── utils/
    ├── __init__.py
    ├── logging.py          # Logging setup (if separate)
    ├── plotting.py         # Functions for plotting distributions and molecules
    └── general.py          # General utilities like sample_distribution
```


### Overview of the Files

Here is a breakdown of all of the files that are in this repository and what they do. Please reference the chart to understand what files need to be changed to add functionality and get the entire process working. 

| File/Directory Name                   | Description                                                                                                                          |
|:-------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|
| `main.py`                             | **Entry Point**: Orchestrates the entire workflow. Initializes configurations, loads data, sets up models, and starts the training or evaluation process. |
| `config.py`                           | **Configuration**: Centralizes all configuration variables, hyperparameters, flags, and constants used across the project. Modify settings here to change experiment configurations without altering core code. |
| `README.md`                           | **Documentation**: This file provides an overview, setup instructions, and detailed explanations of the project structure and functionalities. |
| `requirements.txt`                    | **Dependencies**: Lists all Python packages required to run the project. Install them using `pip install -r requirements.txt`.       |
| `models/`                             | **Model Implementations**: Contains all model-related code, including the Generator and Discriminator models.                         |
| ├── `__init__.py`                     | **Package Initialization**: Makes the `models` directory a Python package.                                                            |
| ├── `generator.py`                    | **Generator Model**: Implements the Generator model responsible for creating molecular structures using graph neural networks.        |
| ├── `discriminator.py`                | **Discriminator Model**: Implements the Discriminator model that evaluates the validity and quality of generated molecules.            |
| └── `utils.py`                        | **Model Utilities**: Contains utility classes and functions for models, such as residual layers and baseline implementations.          |
| `data/`                               | **Data Handling**: Manages all data-related operations, including loading, processing, and utility functions for molecular data.     |
| ├── `__init__.py`                     | **Package Initialization**: Makes the `data` directory a Python package.                                                              |
| ├── `constants.py`                    | **Constants**: Defines constants like allowed atom types, valence dictionaries, and motif SMARTS patterns used throughout the project. |
| ├── `process.py`                      | **Data Processing**: Contains functions to load datasets, process molecular data, and convert molecules into graph representations suitable for GNNs. |
| └── `utils.py`                        | **Data Utilities**: Implements classes like `MolEncoder` and `MolGraphGeneration` that handle encoding molecular information into graph formats. |
| `rl/`                                 | **Reinforcement Learning**: Implements reinforcement learning components, including rewards and penalties for model training.           |
| ├── `__init__.py`                     | **Package Initialization**: Makes the `rl` directory a Python package.                                                                 |
| ├── `rewards.py`                      | **Rewards**: Defines functions to compute various rewards based on the properties and performance of generated molecules.               |
| ├── `penalties.py`                    | **Penalties**: Contains functions to compute penalties that discourage the generation of undesirable molecular features.               |
| └── `utils.py`                        | **RL Utilities**: Provides utility functions for reinforcement learning, such as applying flags to rewards and penalties.                |
| `utils/`                              | **General Utilities**: Offers general-purpose utilities used across the project, including logging, plotting, and helper functions.    |
| ├── `__init__.py`                     | **Package Initialization**: Makes the `utils` directory a Python package.                                                              |
| ├── `logging.py`                      | **Logging Setup**: Configures logging to track the progress and debug the training and evaluation processes.                           |
| ├── `plotting.py`                     | **Plotting Functions**: Provides functions to plot distributions of molecular properties, visualize generated molecules, and create other relevant plots. |
| └── `general.py`                      | **General Utilities**: Implements utility functions like sampling from distributions and other helper methods used across the project. |
| `scripts/`                            | **Automation Scripts**: Contains various scripts for dataset creation, model training, evaluation, and visualization.                  |
| ├── `run_attn_model.sh`                | **HPC Submission Script**: Shell script optimized for submitting training jobs to the Yale HPC system. Can be modified for other high-performance computing environments. |
| ├── `visualize_results.py`             | **Visualization Script**: Script dedicated to visualizing training results, including loss curves and generated molecular structures.  |
| ├── `creating_ligand_dataset.py`       | **Dataset Creation**: Script to create and preprocess the ligand dataset from raw data sources.                                       |
| ├── `train_model.py`                   | **Model Training**: Handles the training loop, loss computations, and model checkpointing using the prepared dataset.                  |
| └── `evaluate_model.py`                | **Model Evaluation**: Assesses the trained models' performance on benchmark molecules and computes relevant metrics.                    |
| `data_files/`                         | **Data Files**: Stores all data files required for training and testing.                                                              |
| ├── `training_molecules_filepaths.txt` | **Training Data Paths**: Lists all file paths to protein-ligand combinations used to train the machine learning model.                  |
| └── `testing_molecules_filepaths.txt`  | **Testing Data Paths**: Lists all file paths to protein-ligand combinations used to test and evaluate the model's performance.           |
| `results/`                            | **Results and Outputs**: Directory to store all outputs from training and evaluation, including models, logs, and images.              |
| ├── `models/`                          | **Saved Models**: Contains saved model checkpoints from different training epochs.                                                    |
| ├── `logs/`                            | **Log Files**: Stores log files capturing detailed information about the training and evaluation processes.                           |
| └── `images/`                          | **Generated Images**: Holds generated images of molecules for visualization and analysis.                                             |

