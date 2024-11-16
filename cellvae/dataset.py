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
        if os.path.exists('data/thumbnails'):
            logging.info('Thumbnails loaded.')
        else:
            logging.info('Creating thumbnails.')
            self.create_thumbnails()
    
    def create_thumbnails(self):
        """Create thumbnails."""
        os.makedirs('data/thumbnails')
        csv = pd.read_csv(self.config.data.csv, usecols=self.config.data.csv_xy)
        csv = csv.to_numpy(dtype='int')
        img = tiff.imread(self.config.data.img)
        img = np.moveaxis(img, 2, 0) # temp
        cropper = ImageCropper(self.config, img, csv)
        cropper.crop()
    def load_labels(self):
        """Load labels."""
        labels = pd.read_csv(self.config.data.csv, usecols=self.config.data.csv_labels)
        assert len(labels) == self.__len__()
        self.labels = torch.tensor(labels.to_numpy().flatten())
        pass

    def __len__(self):
        files = os.listdir('data/thumbnails')
        return len(files)
    
    def __getitem__(self, idx):
        path = f'data/thumbnails/cell_{idx}.tif'
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
        """Load or create split indices."""
        logging.info('Checking for train/test split...')
        try:
            self.train_idx = np.load('data/train_idx.npy')
            logging.info('Split loaded.')
        except FileNotFoundError:
            logging.info('Creating new split.')
            self.create_split()
        self.valid_idx = np.setdiff1d(np.arange(len(self.dataset)), self.train_idx)

    def create_split(self):
        """Create split indices."""
        train_size = int(self.config.train.train_p * len(self.dataset))
        idx = np.arange(len(self.dataset))
        self.train_idx = np.random.choice(idx, train_size, replace=False)
        np.save('data/train_idx.npy', self.train_idx)
    
    def apply_split(self):
        self.train_set = Subset(self.dataset, self.train_idx)
        self.valid_set = Subset(self.dataset, self.valid_idx)
        self.train_loader = self.load_dataset(self.train_set)
        self.valid_loader = self.load_dataset(self.valid_set)
    
    def load_dataset(self, dataset):
        if len(dataset) == 0:
            return None
        return DataLoader(dataset,
                          batch_size=self.config.train.batch_size,
                          num_workers=self.config.train.num_workers,
                          shuffle=True)
