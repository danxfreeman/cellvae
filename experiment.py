import wandb

from cellvae.utils import load_config, init_log
from cellvae.dataset import CellLoader
from cellvae.agent import CellAgent

def init_wanb(config):
    wandb.init(
        project='CellCNN',
        name='dropout',
        notes='set dropout to 0.5',
        config=config
    )

def main():
    config = load_config()
    init_log()
    init_wanb(config)
    loader = CellLoader(config)
    agent = CellAgent(config, loader)
    agent.run()
    wandb.finish()

if __name__ == '__main__':
    main()
