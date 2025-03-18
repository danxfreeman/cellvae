import os
import logging

logging.getLogger().setLevel(logging.INFO)

import tifffile as tiff
import pandas as pd
import numpy as np

class CellCropper():

    def __init__(self, config, outdir='data', batch_size=1000):
        self.config = config
        self.outdir = outdir
        self.batch_size = batch_size
        self.window = config.preprocess.crop_size
        self.offset = self.window // 2
        self.load_img()
        self.load_csv()
    
    def run(self):
        """Prepare dataset."""
        os.makedirs(self.outdir, exist_ok=True)
        self.filter_cells()
        if self.config.data.csv_subset:
            self.subset_cells()
        np.save(f'{self.outdir}/subset_idx.npy', self.csv.index.to_numpy())
        self.split_cells()
        self.crop_cells()
    
    def load_img(self):
        """Load image."""
        logging.info('Loading image...')
        self.img = tiff.imread(self.config.data.img).transpose(1, 2, 0)
        logging.info('Image loaded.')
    
    def load_csv(self):
        """Load metadata."""
        cols = dict(zip(self.config.data.csv_xy, ['x', 'y']))
        self.csv = pd.read_csv(self.config.data.csv, usecols=cols.keys()).rename(columns=cols).astype(int)
    
    def filter_cells(self):
        """Filter cells near image boundaries."""
        img_height, img_width, _ = self.img.shape
        self.csv = self.csv[
            (self.csv.x > self.offset) & (self.csv.x < img_width - self.offset) &
            (self.csv.y > self.offset) & (self.csv.y < img_height - self.offset)
        ]
    
    def subset_cells(self):
        """Create random subset."""
        self.csv = self.csv.sample(n=self.config.data.csv_subset).sort_index()
    
    def split_cells(self):
        """Create random test/train split."""
        train_size = int(self.config.train.train_ratio * len(self.csv))
        train_idx = np.sort(np.random.choice(len(self.csv), size=train_size, replace=False))
        valid_idx = np.setdiff1d(np.arange(len(self.csv)), train_idx)
        np.save(f'{self.outdir}/valid_idx.npy', valid_idx)
        np.save(f'{self.outdir}/train_idx.npy', train_idx)

    def crop_cells(self):
        """Create cell thumbnails."""
        thumbnails = np.zeros((len(self.csv), self.window, self.window, 3), dtype=np.uint8)
        for i, x in enumerate(self.crop()):
            thumbnails[i] = x
        logging.info('Exporting thumbnails...')
        np.save(f'{self.outdir}/thumbnails.npy', thumbnails)
        logging.info('Thumbnails exported.')
    
    def crop(self):
        """Crop cells."""
        for idx, (xcenter, ycenter) in enumerate(self.csv[['x', 'y']].values):
            if idx % 10000 == 0:
                logging.info(f'Cropping cell {idx} of {len(self.csv)}.')
            xstart, xend = xcenter - self.offset, xcenter + self.offset
            ystart, yend = ycenter - self.offset, ycenter + self.offset
            yield self.img[ystart:yend, xstart:xend]

class InferenceCropper(CellCropper):

    def __init__(self, config, img, csv, outdir='test_data', batch_size=1000):
        super().__init__(config, outdir, batch_size)
        self.img = img
        self.csv = csv.reset_index(drop=True)
        self.csv.columns = ['x', 'y'] + self.csv.columns[2:].to_list()
    
    def load_img(self):
        """Do not load data."""
        pass

    def load_csv(self):
        """Do not load metadata."""
        pass

    def split_cells(self):
        """Do not split cells."""
        pass

    def subset_cells(self):
        """Do not subset cells."""
        pass

if __name__ == '__main__':
    from cellvae.utils import load_config
    config = load_config()
    cropper = CellCropper(config)
    cropper.run()
