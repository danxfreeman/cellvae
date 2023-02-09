# Import modules.
import os
import json
import logging
import pandas as pd
from easydict import EasyDict

# Import configuration file.
def load_config(json_file):
    with open(json_file, 'r') as f:
        try:
            config_dict = json.load(f)
            config = EasyDict(config_dict)
            config = autofill_config(config)
            return config
        except ValueError:
            print('Invalid JSON file format')

# Autofill configuration file based on data.
def autofill_config(config):
    markers = pd.read_csv(config.input.markers)
    config.input.n_channels = len(markers)
    config.input.channel_name = list(markers.marker_name)
    config.input.channel_number = list(markers.channel_number)
    return config

# Initialize logger.
def setup(config):
    if not os.path.exists(config.input.output):
        os.makedirs(config.input.output)
    config_file = os.path.join(config.input.output, 'config.json')
    with open(config_file, 'w') as fp:
        json.dump(config, fp, indent=4)
    logr_file = os.path.join(config.input.output, 'log.txt')
    logging.basicConfig(filename=logr_file, level=logging.INFO)
    logging.info('****Initializing experiment****')
