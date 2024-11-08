from cellvae.utils import load_config, init_log
from cellvae.dataset import CellLoader
from cellvae.agent import CellAgent

config = load_config()
init_log()

if __name__ == '__main__':
    loader = CellLoader(config)
    agent = CellAgent(config, loader)
    agent.run()
