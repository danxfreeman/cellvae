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

# Autofill configuration file.
def autofill_config(config):
    keys = set(['n_channels', 'channel_name', 'channel_number'])
    if not keys.issubset(config.input.keys()):
        markers = pd.read_csv(config.input.markers)
        config.input.n_channels = len(markers)
        config.input.channel_name = list(markers.marker_name)
        config.input.channel_number = list(markers.channel_number)
    return config
