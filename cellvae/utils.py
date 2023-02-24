# Import modules.
import json
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
    if not config.input.n_channels:
        config.input.n_channels = len(markers)
    if not config.input.channel_name:
        config.input.channel_name = list(markers.marker_name)
    if not config.input.channel_number:
        config.input.channel_number = list(markers.channel_number)
    return config
