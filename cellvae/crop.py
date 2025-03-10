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
    
    def load_img(self):
        """Load image."""
        logging.info('Loading image...')
        self.img = tiff.imread(self.config.data.img).transpose(1, 2, 0)
        logging.info('Image loaded.')
    
    def load_csv(self):
        """Load centroids."""
        cols = {self.config.data.csv_xy[0]: 'x', self.config.data.csv_xy[1]: 'y'}
        csv = pd.read_csv(self.config.data.csv, usecols=cols.keys()).rename(columns=cols)
        img_height, img_width, _ = self.img.shape
        self.csv = csv[
            (csv.x > self.offset) & (csv.x < img_width - self.offset) &
            (csv.y > self.offset) & (csv.y < img_height - self.offset)
        ].to_numpy()

    def run(self):
        """Prepare dataset."""
        thumbnails = np.stack(list(self.crop()))
        logging.info('Exporting...')
        os.makedirs('data/', exist_ok=True)
        np.save('data/thumbnails.npy', thumbnails)
        logging.info('Thumbnails exported.')
    
    def crop(self):
        """Crop cell thumbnails."""
        for idx, (xcenter, ycenter) in enumerate(self.csv):
            if idx % 10000 == 0:
                logging.info(f'Cropping cell {idx} of {len(self.csv)}.')
            xstart, xend = xcenter - self.offset, xcenter + self.offset
            ystart, yend = ycenter - self.offset, ycenter + self.offset
            yield self.img[ystart:yend, xstart:xend]

if __name__ == '__main__':
    from cellvae.utils import load_config
    config = load_config()
    cropper = CellCropper(config)
    cropper.run()
