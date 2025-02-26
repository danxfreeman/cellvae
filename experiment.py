from cellvae.utils import load_config, init_log, init_wanb
from cellvae.dataset import CellLoader
from cellvae.agent import CellAgent

def main():
    config = load_config()
    init_log()
    init_wanb()
    loader = CellLoader(config)
    agent = CellAgent(config, loader)
    agent.run()

if __name__ == '__main__':
    main()
