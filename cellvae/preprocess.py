import logging

import numpy as np

import tifffile as tiff

class ImageCropper:

    def __init__(self, config, img, csv):
        self.crop_size = config.preprocess.crop_size
        self.img = img
        self.csv = csv
        self.total = len(self.csv)
        self.cropped = 0
    
    def crop(self):
        """Crop all cells."""
        for idx, (xcenter, ycenter) in enumerate(self.csv):
            self.crop_cell(xcenter, ycenter)
            if idx % 100 == 0:
                logging.info(f'Cropping cell {idx} of {self.total}.')
    
    def crop_cell(self, xcenter, ycenter):
        """Crop one cell."""
        expand = self.crop_size // 2
        xstart = xcenter - expand
        ystart = ycenter - expand
        xend = xstart + self.crop_size
        yend = ystart + self.crop_size
        if xstart >= 0 and ystart >= 0 and xend <= self.img.shape[1] and yend <= self.img.shape[2]:
            thumbnail = self.img[:, xstart:xend, ystart:yend]
            tiff.imwrite(f'data/thumbnails/cell_{self.cropped}.tif', data=thumbnail)
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
