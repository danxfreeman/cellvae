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
        os.makedirs('data/', exist_ok=True)
        self.load_img()
        self.load_csv()
        self.filter_cells()
    
    def load_img(self):
        """Load image."""
        logging.info('Loading image...')
        self.img = tiff.imread(self.config.data.img).transpose(1, 2, 0)
        logging.info('Image loaded.')
    
    def load_csv(self):
        """Load cell data."""
        cols = {self.config.data.csv_xy[0]: 'x', self.config.data.csv_xy[1]: 'y'}
        self.csv = pd.read_csv(self.config.data.csv, usecols=cols.keys()).rename(columns=cols)
    
    def filter_cells(self):
        """Filter cells near image boundaries."""
        img_height, img_width, _ = self.img.shape
        self.csv = self.csv[
            (self.csv.x > self.offset) & (self.csv.x < img_width - self.offset) &
            (self.csv.y > self.offset) & (self.csv.y < img_height - self.offset)
        ]
        if self.config.data.csv_subset:
            self.csv = self.csv.sample(self.config.data.csv_subset)
            np.save('data/subset_idx.npy', self.csv.index.to_numpy())
        self.csv = self.csv.to_numpy(dtype=int)

    def run(self):
        """Prepare dataset."""
        thumbnails = np.stack(list(self.crop()))
        train_idx, valid_idx = self.split(size=len(thumbnails))
        logging.info('Exporting...')
        np.save('data/thumbnails.npy', thumbnails)
        np.save('data/valid_idx.npy', valid_idx)
        np.save('data/train_idx.npy', train_idx)
        logging.info('Thumbnails exported.')
    
    def crop(self):
        """Crop cell thumbnails."""
        for idx, (xcenter, ycenter) in enumerate(self.csv):
            if idx % 10000 == 0:
                logging.info(f'Cropping cell {idx} of {len(self.csv)}.')
            xstart, xend = xcenter - self.offset, xcenter + self.offset
            ystart, yend = ycenter - self.offset, ycenter + self.offset
            yield self.img[ystart:yend, xstart:xend]
    
    def split(self, size):
        """Apply random test/train split."""
        indices = np.random.permutation(size)
        train_size = int(self.config.train.train_ratio * size)
        return np.sort(indices[:train_size]), np.sort(indices[train_size:])

if __name__ == '__main__':
    from cellvae.utils import load_config
    config = load_config()
    cropper = CellCropper(config)
    cropper.run()
