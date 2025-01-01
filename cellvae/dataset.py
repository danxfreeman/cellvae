import os
import logging

import torch
import numpy as np
import pandas as pd
import tifffile as tiff

from torch.utils.data import DataLoader, Dataset, Subset

from cellvae.preprocess import ImageCropper, Vignette

class CellDataset(Dataset):

    def __init__(self, config):
        self.config = config
        self.vignette = Vignette(self.config)
        self.load_thumbnails()
        self.load_labels()
    
    def load_thumbnails(self):
        """Load or create thumbnails."""
        logging.info('Checking for thumbnails...')
        if os.path.exists(self.config.data.thumbnails):
            logging.info('Thumbnails loaded.')
        else:
            logging.info('Creating thumbnails.')
            cropper = ImageCropper(self.config)
            cropper.crop()
    
    def load_labels(self):
        """Load labels."""
        labels = pd.read_csv(self.config.data.csv, usecols=self.config.data.csv_labels)
        labels = labels.to_numpy()
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        path = f'{self.config.data.thumbnails}/cell_{idx}.tif'
        thumbnail = tiff.imread(path)
        thumbnail = self.vignette(thumbnail)
        return torch.Tensor(thumbnail), self.labels[idx]

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
