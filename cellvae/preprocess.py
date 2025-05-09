import os
import logging

import numpy as np

class CellCropper():
    
    def __init__(self, img, csv, outdir='data', crop_size=40, batch_size=1_000):
        self.img = img
        self.csv = csv.reset_index(drop=True).astype(int)
        self.csv.columns = ['x', 'y', 'lab']
        self.outdir = outdir
        self.crop_size = crop_size
        self.offset = self.crop_size // 2
        self.batch_size = batch_size
        os.makedirs(self.outdir, exist_ok=True)

    def crop(self):
        """Prepare dataset."""
        self.filter_cells()
        np.save(f'{self.outdir}/subset_idx.npy', self.csv.index.to_numpy())
        np.save(f'{self.outdir}/labels.npy', self.csv.lab.to_numpy())
        self.crop_cells()
        logging.info('Exporting thumbnails.')
        np.save(f'{self.outdir}/thumbnails.npy', self.thumbnails)
        logging.info('Done.')
    
    def split(self):
        """Create balanced test/train split."""
        train_idx = []
        for lab in [0, 1]:
            indices = np.where(self.csv['lab'] == lab)[0]
            size = int(len(indices) * self.config.train.train_ratio)
            train_idx.extend(np.random.choice(indices, size=size, replace=False))
        train_idx = np.sort(train_idx)
        valid_idx = np.setdiff1d(np.arange(len(self.csv)), train_idx)
        np.save(f'{self.outdir}/valid_idx.npy', valid_idx)
        np.save(f'{self.outdir}/train_idx.npy', train_idx)

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

