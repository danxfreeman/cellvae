import json

import wandb

from easydict import EasyDict

def load_config(config_path='config.json'):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        return EasyDict(config_dict)

def init_wandb(track=False, project='CellVAE', **kwargs):
    """Initialize remote tracking. Must call `wandb.finish()` after the run."""
    mode = 'online' if track else 'disabled'
    wandb.init(
        project=project,
        mode=mode,
        **kwargs
    )

