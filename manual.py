import os

import pandas as pd

from cellvae.utils import load_config
from cellvae.dataset import CellDataset
from cellvae.plot import Plot

# Import dataset.
config = load_config()
dataset = CellDataset(config)
plot = Plot(config, dataset)

# Import or initialize responses.
OUT_PATH = '/n/scratch/users/d/daf179/melanoma/ML/responses.csv'
if os.path.exists(OUT_PATH):
    response = pd.read_csv(OUT_PATH)
else:
    response = pd.DataFrame(columns=['cell_id', 'response'])
    response.to_csv(OUT_PATH, index=False)
print(f'Loaded {len(response)} responses.')

# Subset ids.
labels = dataset.csv.copy()
labels = labels[~labels.id.isin(response.cell_id)]
labels = labels.groupby('label').sample(n=1000).sample(frac=1).reset_index(drop=True)

# Label cells.
yes = 0
for i, id_ in enumerate(labels.id):
    plot.cell(id_, window=224, outline=False)
    response = input('Is this cell mitotic (y/[n])?')
    response = 'y' if response == 'y' else 'n'
    yes += response == 'y'
    df = pd.DataFrame([{'cell_id': id_, 'response': response}])
    df.to_csv(OUT_PATH, mode='a', header=False, index=False)
    if i % 5 == 0:
        print(f'{i} new responses ({yes} mitotic)')

