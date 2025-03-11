import json

from easydict import EasyDict

def load_config(config_path='config.json'):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        return EasyDict(config_dict)

