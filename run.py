# Import modules.
import argparse
from utils import load_config
from dataset import CellLoader
from agent import CellAgent

# Create parser.
parser = argparse.ArgumentParser('VAE for cell typing')
parser.add_argument('config', help='path to the config file')
args = parser.parse_args()

# Run experiment.
config = load_config(args.config)
loader = CellLoader(config)
agent = CellAgent(config, loader)
agent.train()
