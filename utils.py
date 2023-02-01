# Import modules.
import json
from easydict import EasyDict

# Import configuration file.
def load_config(json_file):
    with open(json_file, 'r') as f:
        try:
            config_dict = json.load(f)
            config = EasyDict(config_dict)
            return config
        except ValueError:
            print('Invalid JSON file format')
