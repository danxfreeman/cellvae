# CellVAE

Embed multiplex cell thumbnails.

Based on [Morphology-Aware Profiling of Highly Multiplexed Tissue Images using Variational Autoencoders
](https://pmc.ncbi.nlm.nih.gov/articles/PMC12262432/).

## Settings

VAE settings are stored in `config.json`.

### Preprocessing

* `crop_size`: Thumbnail size in pixels
* `alpha`, `beta`: Vignette strength and shape (0 = no vignette)
* `min_quant`, `max_quant`: Quantile values used to clip extreme intensities

### Training

* `train_ratio`: Fraction of thumbnails to use for training
* `batch_size`: Number of samples to process before each update
* `num_epochs`: Number of train iterations over the entire dataset

### Model

* `learning_rate`: How quickly the model learns
* `latent_dim`: Number of features in the bottleneck layer
* `beta`: KL coefficient

Lower `beta` values prioritize accurate reconstructions (MSE), which higher values encourage smooth embeddings (KL).

## Modules

### Preprocessing

`CellCropper` generates a `.npy` file containing pre-transformed thumbnails.

```
from cellvae.utils import load_config
from cellvae.preprocess import CellCropper

config = load_config()

cropper = CellCropper(
    img, # multiplex image (CYX array)
    cells[['X_centroid', 'Y_centroid']], # cell coordinates
    dirname='datadir', # output directory
    config=config, # config object
    transform_fn=None # optional transformation function
)
cropper.crop()
```

Results will be saved to `dirname`.

* `thumbnails.npy`: Cropped thumbnails
* `subset_idx.npy`: Row indices used to make thumbnails. Cropper automatically excludes boundary cells.

Load the entire image into memory for faster cropping. Otherwise, load the image with `zarr`. Manually subset relevant channels and cells before cropping.

### Training

`CellLoader` loads thumbnails and `CellAgent` trains the model.

```
from cellvae.utils import load_config
from cellvae.dataset import CellLoader
from cellvae.agent import CellAgent

config = load_config()

loader = CellLoader(config, dirname='datadir', augment_fn=None)
agent = CellAgent(config, loader, outdir='datadir')
agent.run()
```

Results will be saved to `outdir`.

* `valid_idx.npy`: Thumbnail indices used for validation
* `checkpoint.pth.tar`: Model weights (updates each epoch)
* `loss.csv`: Train and validation losses

### Inference

`CellAgent` automatically loads weights and indices from `datadir`.

```
import numpy as np
import torch

from cellvae.utils import load_config
from cellvae.agent import CellAgent

# Load model.
config = load_config()
agent = CellAgent(config, outdir='datadir', in_channels=3) # must specify `in_channels` if no loader

# Simulate data.
x = torch.rand(100, 3, 32, 32) # can also be a loader

# Encode cells.
embedding = np.concatenate(list(agent.encode(x)), axis=0)

# Reconstruct cells.
decoded = np.concatenate(list(agent.decode(x)), axis=0)
```

### Transformations

CellVAE comes with two transformations.

* `LogTransform`: applies a `log1p` transformation and clips extreme intensity values.
* `Vignette`: attenuates pixels near thumbnail corners and edges.

Transformations can be applied during __cropping__ or __training__.

* __During cropping:__ transformations will be applied once to the entire dataset and saved in `thumbnails.npy`.
* __During training:__ transformations will be applied on the fly each batch.

## MCMICRO Example

All input comes from MCMICRO and `config.json`.

```
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
    dirname='datadir',
    transform_fn=transform
)
cropper.crop()

# Train model.
loader = CellLoader(config, dirname='datadir')
agent = CellAgent(config, loader, outdir='datadir')
agent.run()

# Embed cells.
embedding = np.concatenate(agent.encode(loader), axis=0)
embedding_cols = [f'Z{str(i + 1).zfill(3)}' for i in range(embedding.shape[1])]
embedding = pd.DataFrame(embedding, columns=embedding_cols)
embedding.to_csv('datadir/embedding.csv')
```

See `examples/` for more detailed examples.
