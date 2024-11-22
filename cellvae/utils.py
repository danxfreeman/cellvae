import json
import logging

from easydict import EasyDict

def load_config(config_path='config.json'):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        return EasyDict(config_dict)

def init_log():
    """Initialize log file."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %I:%M:%S %p'
    )
    logging.info('***Initializing experiment***')
