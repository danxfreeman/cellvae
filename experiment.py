import torch
import pandas as pd
from utils import load_config, init_logger
from dataset import CellLoader
from agent import CellAgent
from plotting import Plot

config = load_config('/Users/danfreeman/Dropbox/Jonas/scripts/VAE/config.json')
init_logger(config)
loader = CellLoader(config)
# agent = CellAgent(config, loader)

# idx = list(range(10000))
# newdata = torch.utils.data.Subset(loader.valid_set, idx)
# z = agent.embed(newdata)
# z = pd.DataFrame(z)
# z.to_csv('latent.csv')

markers = pd.read_csv(config.input.markers)
# Plot(config, newdata).plot_thumbnails(
#     cell_index=[0,1,2,3,4],
#     channel_name=[markers.marker[i] for i in [0,1,2,3,4]],
#     channel_index=[0,1,2,3,4],
#     channel_boost=[2,1,1,1,1],
#     filename='temp.pdf'
# )

# idx = list(range(20))
# newdata = torch.utils.data.Subset(loader.valid_set, idx)
# recon = agent.recon(newdata)
# recon = torch.split(recon, 1)
# recon = [x.squeeze() for x in recon]

# Plot(config, recon).plot_thumbnails(
#     cell_index=[0,1,2,3,4],
#     channel_name=[markers.marker[i] for i in [0,1,2,3,4]],
#     channel_index=[0,1,2,3,4],
#     channel_boost=[2,2,2,2,2],
#     filename='temp.pdf'
# )

idx = list(range(100))
newdata = torch.utils.data.Subset(loader.valid_set, idx)
Plot(config, newdata).plot_channels(filename='test.pdf')