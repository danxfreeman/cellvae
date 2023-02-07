# Import modules.
import argparse
from cellvae.utils import load_config, init_logger
from cellvae.dataset import CellLoader
from cellvae.agent import CellAgent

# Create parser.
parser = argparse.ArgumentParser('VAE for cell typing')
parser.add_argument('config', help='path to the config file')
args = parser.parse_args()

# Run experiment.
config = load_config(args.config)
init_logger(config)
loader = CellLoader(config)
agent = CellAgent(config, loader)
agent.train()
