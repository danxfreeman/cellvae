import os
import logging

import torch
import numpy as np
import pandas as pd
import tifffile as tiff

from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

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
        cropped = np.load('data/cropped.npy')
        self.labels = torch.tensor(labels[cropped], dtype=torch.float32)

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
            self.train_idx = np.load('data/train_idx.npy')
            logging.info('Split loaded.')
        except FileNotFoundError:
            logging.info('Creating new split.')
            self.create_split()
        self.valid_idx = np.setdiff1d(np.arange(len(self.dataset)), self.train_idx)

    def create_split(self):
        """Create train/test split indices."""
        train_size = int(self.config.train.train_ratio * len(self.dataset))
        idx = np.arange(len(self.dataset))
        self.train_idx = np.random.choice(idx, train_size, replace=False)
        np.save('data/train_idx.npy', self.train_idx)
    
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
        sample_weights = self.weigh_samples(dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers
        )
    
    def weigh_samples(self, dataset):
        """Balance batch class composition."""
        subset = dataset.indices
        labels = dataset.dataset.labels[subset].flatten().long()
        class_counts = torch.bincount(labels)
        pos_ratio = self.config.train.pos_ratio
        class_ratio = torch.tensor([1 - pos_ratio, pos_ratio])
        return (class_ratio / class_counts)[labels]
