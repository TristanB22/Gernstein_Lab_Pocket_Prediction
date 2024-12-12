# utils/logging.py

'''
Setup and configure logging for the project.

This module provides a function to initialize logging based on configuration flags.
It sets up log handlers for writing logs to files and streaming them to the console.
Additionally, it initializes a separate logger for reinforcement learning (RL) losses
if enabled.
'''

import logging
import os
from config import LOGGING_ENABLED, run_dir, LOG_RL_LOSSES


def setup_logging():
    '''
    Initializes logging configurations based on project settings.

    If logging is enabled, it sets up file handlers for general logs and optionally
    for reinforcement learning losses. If logging is disabled, it defaults to a minimal
    logging level.

    :return: None
    '''
    
    if LOGGING_ENABLED:
        # define file paths for loss logs
        loss_file_path = os.path.join(run_dir, "loss.txt")
        gen_disc_loss_path = os.path.join(run_dir, "gen_loss.txt")
        
        # set logging level to debug
        logging_level = logging.DEBUG
        
        # configure basic logging with file and stream handlers
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(run_dir, "enhanced_training.log")),
                logging.StreamHandler()
            ]
        )
        logging.info("logging is enabled.")
    
        if LOG_RL_LOSSES:
            # define file path for RL loss logs
            rl_loss_file_path = os.path.join(run_dir, "rl_losses.log")
            
            # create a separate logger for RL losses
            rl_logger = logging.getLogger('rl_logger')
            rl_logger.setLevel(logging.INFO)
            
            # create and configure file handler for RL losses
            rl_handler = logging.FileHandler(rl_loss_file_path)
            rl_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            rl_logger.addHandler(rl_handler)
            
            logging.info("rl loss logger initialized.")
    else:
        # configure basic logging with warning level only
        logging.basicConfig(level=logging.WARNING)
        logging.info("logging is minimal.")
