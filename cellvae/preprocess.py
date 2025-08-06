import os
import logging

import numpy as np

class CellCropper():
    
    def __init__(self, img, csv, dirname='data', crop_size=32, batch_size=1_000, transform_fn=None):
        self.img = img
        self.n_channels = self.img.shape[0]
        self.csv = csv.reset_index(drop=True).astype(int)
        self.n_cells = self.csv.shape[0]
        self.dirname = dirname
        self.crop_size = crop_size
        self.offset = self.crop_size // 2
        self.batch_size = batch_size
        self.transform_fn = transform_fn

    def crop(self):
        """Prepare dataset."""
        self.filter_cells()
        self.crop_cells()
        if self.transform_fn:
            self.thumbnails = self.transform_fn(self.thumbnails)
        os.makedirs(self.dirname, exist_ok=True)
        np.save(f'{self.dirname}/subset_idx.npy', self.csv.index.to_numpy())
        np.save(f'{self.dirname}/thumbnails.npy', self.thumbnails)
        logging.info('Done.')

    def filter_cells(self):
        """Filter cells near image boundaries."""
        _, img_height, img_width = self.img.shape
        x, y = self.csv.values.T
        self.csv = self.csv[
            (x > self.offset) & (x < img_width - self.offset) &
            (y > self.offset) & (y < img_height - self.offset)
        ]
    
    def crop_cells(self):
        """Crop all cells."""
        self.thumbnails = np.zeros((self.n_cells, self.n_channels, self.crop_size, self.crop_size), dtype=np.uint16)
        for i, (xcenter, ycenter) in enumerate(self.csv.values):
            self.thumbnails[i] = self.crop_one(xcenter, ycenter)
            if i % 10_000 == 0:
                logging.info(f'Cropping cell {i} of {self.n_cells}.')

    def crop_one(self, xcenter, ycenter):
        """Crop one cell."""
        xstart, xend = xcenter - self.offset, xcenter + self.offset
        ystart, yend = ycenter - self.offset, ycenter + self.offset
        return self.img[:, ystart:yend, xstart:xend]

