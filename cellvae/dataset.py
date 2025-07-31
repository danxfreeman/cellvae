import logging

import numpy as np

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor

class CellDataset(Dataset):

    def __init__(self, dirname='data'):
        self.thumbnails = np.load(f'{dirname}/thumbnails.npy')
        self.transform = ToTensor()

    def __len__(self):
        return len(self.thumbnails)
    
    def __getitem__(self, idx):
        return self.transform(self.thumbnails[idx])

class CellLoader:

    def __init__(self, config, dirname=None):
        self.config = config
        self.dirname = dirname
        self.dataset = CellDataset(dirname=dirname)
        self.split_indices()
        self.split_dataset()

    def split_indices(self):
        """Load split indicies if available."""
        try:
            self.valid_idx = np.load(f'{self.dirname}/valid_idx.npy')
            self.train_idx = np.setdiff1d(np.arange(len(self.dataset)), self.valid_idx)
            logging.info("Loaded test/train split.")
        except FileNotFoundError:
            logging.info("Creating new test/train split.")
            self.create_split()

    def create_split(self):
        """Create random test/train split."""
        indices = np.arange(len(self.dataset.thumbnails))
        np.random.shuffle(indices)
        split = int(len(indices) * self.config.train.train_ratio)
        self.train_idx = np.sort(indices[:split])
        self.valid_idx = np.sort(indices[split:])
        np.save(f'{self.dirname}/valid_idx.npy', self.valid_idx)
    
    def split_dataset(self):
        """Split dataset into train and test sets."""
        self.train_set = Subset(self.dataset, self.train_idx)
        self.valid_set = Subset(self.dataset, self.valid_idx)
        self.train_loader = self.load_dataset(self.train_set)
        self.valid_loader = self.load_dataset(self.valid_set)
    
    def load_dataset(self, dataset):
        """Load dataset into DataLoader."""
        if len(dataset) == 0:
            return []
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers
        )
