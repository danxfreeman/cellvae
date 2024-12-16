import os
import logging

import numpy as np
import pandas as pd
import tifffile as tiff
import dask.array as da

class ImageCropper:

    def __init__(self, config):
        self.config = config
        self.crop_size = self.config.preprocess.crop_size
        self.load_image()
        self.load_centroids()
        self.total = len(self.csv)
        self.cropped = 0
        self.indices = []

    def load_image(self):
        """Load TIFF image."""
        self.img = tiff.imread(self.config.data.img)
    
    def load_centroids(self):
        """Load cell coordinates."""
        self.csv = pd.read_csv(self.config.data.csv, usecols=self.config.data.csv_xy)
        self.csv = self.csv.to_numpy(dtype='int')
    
    def crop(self):
        """Crop cells and save indices."""
        os.makedirs(self.config.data.thumbnails, exist_ok=True)
        self.crop_cells()
        np.save('data/cropped.npy', np.array(self.indices))
    
    def crop_cells(self):
        """Crop cells."""
        for idx, (xcenter, ycenter) in enumerate(self.csv):
            self.crop_one_cell(idx, xcenter, ycenter)
            if idx % 100 == 0:
                logging.info(f'Cropping cell {idx} of {self.total}.')
    
    def crop_one_cell(self, idx, xcenter, ycenter):
        """Crop one cell."""
        expand = self.crop_size // 2
        xstart = xcenter - expand
        ystart = ycenter - expand
        xend = xstart + self.crop_size
        yend = ystart + self.crop_size
            thumbnail = self.img[:, ystart:yend, xstart:xend]
        if xstart >= 0 and ystart >= 0 and xend <= self.img.shape[2] and yend <= self.img.shape[1]:
            tiff.imwrite(f'{self.config.data.thumbnails}/cell_{self.cropped}.tif', data=thumbnail)
            self.indices.append(idx)
            self.cropped += 1

class Vignette:

    def __init__(self, config):
        self.vignette = config.preprocess.vignette
        self.crop_size = config.preprocess.crop_size
        x = np.linspace(-self.vignette, self.vignette, self.crop_size)
        x = np.exp(-x ** 2 / 2)
        self.mask = x[:, None] * x
    
    def __call__(self, x):
        return x * self.mask[None, :, :]
