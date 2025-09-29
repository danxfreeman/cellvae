import numpy as np
import pandas as pd
import tifffile as tiff

from cellvae.preprocess import CellCropper
from cellvae.dataset import CellLoader
from cellvae.agent import CellAgent
from cellvae.transform import LogTransform
from cellvae.utils import load_config

DATADIR = '/Volumes/HITS/lsp-data/cycif-techdev/Dan_Mitosis_20250409/mcmicro/LSP19912/'
IMAGE = 'LSP19912'
CHANNELS = [
    'Hoechst2',
    'A488', 'A555', 'A647',
    'Tubulin', 'AcetylTUB', 'gTUB',
    'LaminA/C', 'LaminB'
]

# Load marker table.
markers_path = f'{DATADIR}/markers.csv'
markers = pd.read_csv(markers_path, usecols=['marker_name'])
marker_indices = pd.Series(np.arange(len(markers)), index=markers['marker_name'])

# Load cell table.
cell_path = f'{DATADIR}/quantification/{IMAGE}--unmicst_nucleiRing.csv'
cells = pd.read_csv(cell_path, usecols=['X_centroid', 'Y_centroid'])

# Load image.
img_path = f'{DATADIR}/registration/{IMAGE}.ome.tif'
ch_idx = marker_indices[CHANNELS]
img = tiff.imread(img_path, key=ch_idx)

# Crop thumbnails.
config = load_config()
transform = LogTransform(config) # save clipped and log-transformed thumbnails
cropper = CellCropper(
    img,
    cells[['X_centroid', 'Y_centroid']],
    config,
    dirname=DATADIR,
    transform_fn=transform
)
cropper.crop()

# Train model.
loader = CellLoader(config, dirname=DATADIR)
agent = CellAgent(config, loader, outdir=DATADIR)
agent.run()

# Embed cells.
embedding = np.concatenate(agent.encode(loader), axis=0)
embedding_cols = [f'Z{str(i + 1).zfill(3)}' for i in range(embedding.shape[1])]
embedding = pd.DataFrame(embedding, columns=embedding_cols)
embedding.to_csv(f'{DATADIR}/embedding.csv')
