import logging

import torch
import numpy as np
import pandas as pd
import tifffile as tiff

from torch.utils.data import DataLoader, Dataset, Subset

class CellDataset(Dataset):

    def __init__(self, config):
        self.config = config
        self.offset = self.config.preprocess.crop_size // 2
        self.load_image()
        self.load_labels()
        self.filter_labels()

    def load_image(self):
        """Load image."""
        self.img = tiff.imread(self.config.data.img).astype('float32')
        self.img /= 255

    def load_labels(self):
        """Load cell information."""
        csv = pd.read_csv(self.config.data.csv)
        labels = csv[self.config.data.csv_labels].to_numpy()
        self.labels = torch.from_numpy(labels).float()
        self.loc = csv[self.config.data.csv_xy].to_numpy(dtype=np.uint16)
    
    def filter_labels(self):
        """Filter cells near image boundaries."""
        _, img_height, img_width = self.img.shape
        inbound = (
            (self.loc[:, 0] > self.offset) &
            (self.loc[:, 0] < img_width - self.offset) &
            (self.loc[:, 1] > self.offset) &
            (self.loc[:, 1] < img_height - self.offset)
        )
        self.loc = self.loc[inbound]
        self.labels = self.labels[inbound]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        xcenter, ycenter = self.loc[idx]
        xstart, xend = xcenter - offset, xcenter + offset
        ystart, yend = ycenter - offset, ycenter + offset
        thumbnail = self.img[:, ystart:yend, xstart:xend]
        return torch.from_numpy(thumbnail), self.labels[idx]

class CellLoader:

    def __init__(self, config):
        self.config = config
        self.dataset = CellDataset(self.config)
        self.load_split()
        self.apply_split()
    
    def load_split(self):
        """Load or create train/test split indices."""
        logging.info('Checking for train/test split...')
        try:
            self.valid_idx = np.load('data/valid_idx.npy')
            logging.info('Split loaded.')
        except FileNotFoundError:
            logging.info('Creating new split.')
            self.create_split()
        self.train_idx = np.setdiff1d(np.arange(len(self.dataset)), self.valid_idx)

    def create_split(self):
        """Create train/test split indices."""
        valid_ratio = 1 - self.config.train.train_ratio
        valid_size = int(valid_ratio * len(self.dataset))
        self.valid_idx = np.random.choice(len(self.dataset), size=valid_size, replace=False)
        np.save('data/valid_idx.npy', self.valid_idx)
    
    def apply_split(self):
        """Apply train/test split indices."""
        self.train_set = Subset(self.dataset, self.train_idx)
        self.valid_set = Subset(self.dataset, self.valid_idx)
        self.train_loader = self.load_dataset(self.train_set)
        self.valid_loader = self.load_dataset(self.valid_set)
    
    def load_dataset(self, dataset):
        """Load dataset into DataLoader."""
        if len(dataset) == 0:
            return None
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers
        )
