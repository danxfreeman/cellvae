import os
import json
import logging

from easydict import EasyDict

def load_config(config_path='config.json'):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        return EasyDict(config_dict)

def init_log(log_file='data/experiment.log'):
    """Initialize log file."""
    os.makedirs('data', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %I:%M:%S %p',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info('***Initializing experiment***')
