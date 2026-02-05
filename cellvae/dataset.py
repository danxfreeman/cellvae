import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

class CellDataset(Dataset):

    def __init__(self, config, augment_fn=None):
        self.thumbnails = np.load(config.data.thumb_path)
        self.augment = augment_fn or torch.as_tensor

    def __len__(self):
        return len(self.thumbnails)

    def __getitem__(self, idx):
        x = self.thumbnails[idx]
        return self.augment(x)

class CellLoader:

    def __init__(self, config, augment_fn=None):
        self.config = config
        self.dataset = CellDataset(config, augment_fn)
        self.split_indices()
        self.split_dataset()

    def split_indices(self):
        """Load or create test/train split."""
        try:
            self.valid_idx = np.load(f'{self.config.data.result_dir}/valid_idx.npy')
            logging.info('Loaded test/train split.')
        except FileNotFoundError:
            self.valid_idx = self.sample_test()
            np.save(f'{self.config.data.result_dir}/valid_idx.npy', self.valid_idx)
        self.train_idx = np.setdiff1d(np.arange(len(self.dataset)), self.valid_idx)
        logging.info(f'{len(self.train_idx)} train, {len(self.valid_idx)} test.')

    def sample_test(self):
        """Subset random test set."""
        valid_ratio = 1 - self.config.train.train_ratio
        valid_size = int(valid_ratio * len(self.dataset))
        return np.random.choice(len(self.dataset), size=valid_size, replace=False)
    
    def split_dataset(self):
        """Split dataset."""
        self.train_set = Subset(self.dataset, self.train_idx)
        self.valid_set = Subset(self.dataset, self.valid_idx)
        self.train_loader = self.load_dataset(self.train_set)
        self.valid_loader = self.load_dataset(self.valid_set)
    
    def load_dataset(self, dataset):
        """Load dataset."""
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.config.train.batch_size
        )
