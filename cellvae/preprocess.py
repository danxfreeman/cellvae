import os
import logging

logging.getLogger().setLevel(logging.INFO)

import tifffile as tiff
import pandas as pd
import numpy as np

class CellCropper():

    def __init__(self, config, batch_size=1000):
        self.config = config
        self.window = config.preprocess.crop_size
        self.offset = self.window // 2
        self.batch_size = batch_size
        self.load_img()
        self.load_csv()
        self.filter_cells()
    
    def run(self):
        """Prepare dataset."""
        os.makedirs('data/', exist_ok=True)
        if self.config.data.csv_subset:
            self.subset_cells()
        self.split_cells()
        self.crop_cells()
        np.save('data/labels.npy', self.csv.lab.to_numpy())
    
    def load_img(self):
        """Load image."""
        logging.info('Loading image...')
        self.img = tiff.imread(self.config.data.img).transpose(1, 2, 0)
        logging.info('Image loaded.')
    
    def load_csv(self):
        """Load metadata."""
        cols = {
            self.config.data.csv_xy[0]: 'x',
            self.config.data.csv_xy[1]: 'y',
            self.config.data.csv_label: 'lab'
        }
        self.csv = pd.read_csv(self.config.data.csv, usecols=cols.keys()).rename(columns=cols)
    
    def filter_cells(self):
        """Filter cells near image boundaries."""
        img_height, img_width, _ = self.img.shape
        self.csv = self.csv[
            (self.csv.x > self.offset) & (self.csv.x < img_width - self.offset) &
            (self.csv.y > self.offset) & (self.csv.y < img_height - self.offset)
        ]
        self.csv.reset_index(inplace=True)
    
    def subset_cells(self):
        """Create random subset."""
        self.csv = self.csv.sample(frac=1).groupby('lab').head(self.config.data.csv_subset).sort_index()
        np.save('data/subset_idx.npy', self.csv.index.to_numpy())
        self.csv.reset_index(inplace=True)
    
    def split_cells(self):
        """Create balanced test/train split."""
        train_df = self.csv.groupby('lab').sample(frac=self.config.train.train_ratio).sort_index()
        train_idx = train_df.index.to_numpy()
        valid_idx = np.setdiff1d(np.arange(len(self.csv)), train_idx)
        np.save('data/valid_idx.npy', valid_idx)
        np.save('data/train_idx.npy', train_idx)

    def crop_cells(self):
        """Create cell thumbnails."""
        thumbnails = np.zeros((len(self.csv), self.window, self.window, 3))
        for i, x in enumerate(self.crop()):
            thumbnails[i] = x
        logging.info('Exporting thumbnails...')
        np.save('data/thumbnails.npy', thumbnails)
        logging.info('Thumbnails exported.')
    
    def crop(self):
        """Crop cells."""
        for idx, row in self.csv.iterrows():
            if idx % 10000 == 0:
                logging.info(f'Cropping cell {idx} of {len(self.csv)}.')
            xcenter, ycenter = row['x'], row['y']
            xstart, xend = xcenter - self.offset, xcenter + self.offset
            ystart, yend = ycenter - self.offset, ycenter + self.offset
            yield self.img[ystart:yend, xstart:xend]

if __name__ == '__main__':
    from cellvae.utils import load_config
    config = load_config()
    cropper = CellCropper(config)
    cropper.run()
