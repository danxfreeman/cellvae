import os
import logging

import numpy as np

class CellCropper():
    
    def __init__(self, img, csv, dirname='data', crop_size=64, batch_size=1_000):
        self.img = img
        self.csv = csv.reset_index(drop=True).astype(int)
        self.csv.columns = ['x', 'y', 'lab']
        self.outdir = dirname
        self.crop_size = crop_size
        self.offset = self.crop_size // 2
        self.batch_size = batch_size

    def crop(self):
        """Prepare dataset."""
        self.filter_cells()
        self.crop_cells()
        os.makedirs(self.dirname, exist_ok=True)
        np.save(f'{self.dirname}/subset_idx.npy', self.csv.index.to_numpy())
        np.save(f'{self.dirname}/labels.npy', self.csv.lab.to_numpy())
        np.save(f'{self.dirname}/thumbnails.npy', self.thumbnails)
        logging.info('Done.')

    def filter_cells(self):
        """Filter cells near image boundaries."""
        img_height, img_width, _ = self.img.shape
        self.csv = self.csv[
            (self.csv.x > self.offset) & (self.csv.x < img_width - self.offset) &
            (self.csv.y > self.offset) & (self.csv.y < img_height - self.offset)
        ]
    
    def crop_cells(self):
        """Crop all cells."""
        self.thumbnails = np.zeros((len(self.csv), self.crop_size, self.crop_size, 3), dtype=np.uint8)
        for i, (xcenter, ycenter) in enumerate(self.csv[['x', 'y']].values):
            self.thumbnails[i] = self.crop_one(xcenter, ycenter)
            if i % 10_000 == 0:
                logging.info(f'Cropping cell {i} of {len(self.csv)}.')

    def crop_one(self, xcenter, ycenter):
        """Crop one cell."""
        xstart, xend = xcenter - self.offset, xcenter + self.offset
        ystart, yend = ycenter - self.offset, ycenter + self.offset
        return self.img[ystart:yend, xstart:xend]
