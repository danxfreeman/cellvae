import os
import logging

import numpy as np

class CellCropper():

    def __init__(self, img, csv, dirname='data', crop_size=32, transform_fn=None):
        self.img = img
        self.csv = np.asarray(csv, dtype=int)
        self.dirname = dirname
        self.crop_size = crop_size
        self.offset = crop_size // 2
        self.transform_fn = transform_fn
        self.clamp_cells()
    
    def clamp_cells(self):
        """Clamp cells to image boundaries."""
        bounds = np.array(self.img.shape[1:]) - self.offset
        self.csv = self.csv.clip(self.offset, bounds)

    def crop(self):
        """Main function."""
        self.crop_cells()
        if self.transform_fn:
            self.thumbnails = self.transform_fn(self.thumbnails)
        os.makedirs(self.dirname, exist_ok=True)
        np.save(f'{self.dirname}/thumbnails.npy', self.thumbnails)
        logging.info('DONE.')

    def crop_cells(self):
        """Crop all cells."""
        n_cells = len(self.csv)
        n_channels = self.img.shape[0]
        self.thumbnails = np.zeros((n_cells, n_channels, self.crop_size, self.crop_size), dtype=np.uint16)
        for i, (xcenter, ycenter) in enumerate(self.csv):
            self.thumbnails[i] = self.crop_one(xcenter, ycenter)
            if i % 10_000 == 0 or i == n_cells - 1:
                logging.info(f"Cropping cell {i + 1} of {n_cells}.")

    def crop_one(self, xcenter, ycenter):
        """Crop one cell."""
        xstart, xend = xcenter - self.offset, xcenter + self.offset
        ystart, yend = ycenter - self.offset, ycenter + self.offset
        return self.img[:, ystart:yend, xstart:xend]
