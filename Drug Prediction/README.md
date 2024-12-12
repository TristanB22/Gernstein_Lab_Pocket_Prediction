# Gernstein Lab Pocket Prediction

This project is a collaboration between myself [Tristan Brigham](mailto:tristan.brigham@yale.edu) and [Alan Ianeselli](mailto:alan.ianeselli@yale.edu) in the Gerstein Lab at Yale University.


### Purpose of the Program

This project looks to use machine learning to predict candidate drugs for a given ligand/pocket for the improved synthesis and implementation of drugs for medical care. 


### Directory Structure
```
home/
│
├── main.py                 # Entry point to run the entire project
├── config.py               # Central file for hyperparameters and settings
├── submit_generate_molecules.sh # Run the training script
├── README.md               # This documentation
├── environment.yml        # Python dependencies
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

This repository provides a pipeline for training and generating molecular structures using graph neural networks and reinforcement learning. It has been organized into separate directories to maintain clarity and modularity.

| File/Directory Name                   | Description                                                                                                                          |
|:-------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------:|
| `main.py`                             | **Entry Point**: Orchestrates the entire workflow. Initializes configurations, loads data (PDBbind or MOSES), sets up models, and starts the training or evaluation process. Run it with `python main.py --dataset [pdbbind/moses]`. |
| `config.py`                           | **Configuration**: Centralizes configuration variables, hyperparameters, flags, and constants. Modify this file to change experiment settings (e.g., choosing which dataset is used, hyperparameters, etc.). |
| `submit_generate_molecules.sh`        | **HPC Submission Script**: A shell script optimized for submitting the training job to a high-performance computing (HPC) environment. It sets up the conda environment from `environment.yml` and runs `main.py` on the cluster. |
| `README.md`                           | **Documentation**: Provides an overview, setup instructions, and explains how all files and directories relate. |
| `environment.yml`                     | **Environment Definition**: Specifies all Python dependencies and their versions. Create or update the conda environment using `conda env create -f environment.yml`. |
| `models/`                             | **Model Implementations**: Contains model-related code, including the Generator and Discriminator models. |
| ├── `__init__.py`                     | **Package Initialization**: Makes the `models` directory a Python package. |
| ├── `generator.py`                    | **Generator Model**: Implements the generator responsible for creating molecular structures via graph neural networks. |
| ├── `discriminator.py`                | **Discriminator Model**: Implements the discriminator that evaluates the validity and quality of generated molecules. |
| └── `utils.py`                        | **Model Utilities**: Contains utility classes and functions for models, such as residual layers and baseline implementations. |
| `data/`                               | **Data Handling**: Manages data loading, processing, and conversion of molecules into graph representations. |
| ├── `__init__.py`                     | **Package Initialization**: Makes the `data` directory a Python package. |
| ├── `constants.py`                    | **Constants**: Defines allowed atom types, valence dictionaries, and motif SMARTS patterns. |
| ├── `process.py`                      | **Data Processing**: Loads and processes datasets (PDBbind or MOSES), and converts molecules into graph formats suitable for GNNs. |
| └── `utils.py`                        | **Data Utilities**: Implements classes like `MolEncoder` and `MolGraphGeneration` that encode molecular info into graphs. |
| `rl/`                                 | **Reinforcement Learning**: Provides RL components, including reward and penalty calculations. |
| ├── `__init__.py`                     | **Package Initialization**: Makes the `rl` directory a Python package. |
| ├── `rewards.py`                      | **Rewards**: Functions to compute various rewards based on generated molecules' properties. |
| ├── `penalties.py`                    | **Penalties**: Functions to compute penalties that discourage undesirable molecular features. |
| └── `utils.py`                        | **RL Utilities**: Utility functions for reinforcement learning, such as applying flags to rewards and penalties. |
| `utils/`                              | **General Utilities**: Offers general-purpose utilities like logging setup, plotting, and helper functions. |
| ├── `__init__.py`                     | **Package Initialization**: Makes the `utils` directory a Python package. |
| ├── `logging.py`                      | **Logging Setup**: Configures logging for progress and debugging. |
| ├── `plotting.py`                     | **Plotting Functions**: Functions to visualize distributions, generated molecules, and other results. |
| └── `general.py`                      | **General Utilities**: Helper methods like sampling from distributions and other reusable functions. |

---

### Setting Up the Environment

Use the `environment.yml` file to create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate test_run_env_583_final
```

If the environment already exists, it will be updated accordingly. Ensure all dependencies are correctly installed. Alternatively, you can just run the submit script and it will automatically be done.

The data for pdbbind after cleaning can be found [here](https://drive.google.com/drive/folders/1MCen_OiP9Ey2ClrV8CYHIERdZ205X9Ui?usp=sharing).



### Data Preparation

You can use either the **PDBbind** or **MOSES** dataset:

- **PDBbind**: Download the dataset from [PDBbind](http://www.pdbbind.org.cn/), place the `.sdf` files in the directory specified in `config.py` (`refined_set_dir`), and ensure `main.py --dataset pdbbind` is used.

- **MOSES**: Download the MOSES dataset from [MOSES](https://github.com/molecularsets/moses), place the SMILES CSV file as specified in `config.py` (`moses_csv_path`), and run `main.py --dataset moses`.

Make sure to update `config.py` to point to the correct dataset directories and files.



### Running the Training

You have two main options:

1. **Local Training**: Simply run:
   ```bash
   python main.py --dataset [pdbbind or moses]
   ```
   Adjust `config.py` for hyperparameters like batch size, learning rate, and number of epochs.

2. **HPC Submission**: Use the provided SLURM script `submit_generate_molecules.sh`. This script:
   - Loads and/or creates the conda environment using `environment.yml`
   - Submits the job to a GPU-enabled queue
   - Runs `main.py --dataset pdbbind` (or modify it to `moses`)

   Submit the job as:
   ```bash
   sbatch submit_generate_molecules.sh
   ```

   Check the `runs/logs` directory for `.out` and `.err` files that log the progress.



### Changing Hyperparameters

All model hyperparameters and dataset paths are defined in `config.py`. To switch datasets, update `config.py` and use the `--dataset` flag. For example, to train on MOSES dataset:
```bash
python main.py --dataset moses
```

If you want to modify the architecture (e.g., number of GNN layers, hidden dimensions) or reinforcement learning flags, do so directly in `config.py` before running the training script.



### Viewing Results and Logs

- Training logs and metrics are saved in the `runs/` directory.
- Generated model checkpoints (`generator_latest.pth`, `discriminator_latest.pth`) are also stored in `runs/`.
- Visualizations of generated molecules and property distributions are saved under `runs/images/`.

You can modify plotting or logging settings in `config.py` and related utility scripts to control output verbosity and graphical output.

