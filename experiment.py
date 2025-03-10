import logging

logging.getLogger().setLevel(logging.INFO)

from cellvae.utils import load_config, init_wandb
from cellvae.dataset import CellLoader
from cellvae.agent import CellAgent

def main():
    config = load_config()
    init_wandb()
    loader = CellLoader(config)
    agent = CellAgent(config, loader)
    agent.run()

if __name__ == '__main__':
    main()
