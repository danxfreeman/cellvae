import logging

import torch
import numpy as np
import pandas as pd
import tifffile as tiff
import zarr

from torch.utils.data import DataLoader, Dataset, Subset

class CellDataset(Dataset):

    def __init__(self, config):
        self.config = config
        self.offset = self.config.preprocess.crop_size // 2
        self.load_data()
    
    def load_data(self):
        """Load dataset."""
        logging.info('Loading image...')
        self.load_image()
        logging.info('Loading labels...')
        self.load_labels()
        self.filter_labels()
        logging.info('Loading embeddings...')
        self.load_embeddings()
        logging.info('Loaded.')

    def load_image(self):
        """Load image."""
        if self.config.data.inmemory:
            img = tiff.imread(self.config.data.img)
            self.img = self.transform(img)
        else:
            store = tiff.imread(self.config.data.img, aszarr=True)
            self.img = zarr.open(store, mode='r')[0]

    def load_labels(self):
        """Load cell information."""
        cols = {
            self.config.data.csv_id[0]: 'id',
            self.config.data.csv_xy[0]: 'x',
            self.config.data.csv_xy[1]: 'y',
            self.config.data.csv_labels[0]: 'label'
        }
        self.csv = pd.read_csv(self.config.data.csv, usecols=cols.keys()).rename(columns=cols)
        self.csv.label = self.csv.label.astype(np.float32)
    
    def filter_labels(self):
        """Filter cells near image boundaries."""
        _, img_height, img_width = self.img.shape
        self.csv = self.csv[
            (self.csv.x > self.offset) &
            (self.csv.x < img_width - self.offset) &
            (self.csv.y > self.offset) &
            (self.csv.y < img_height - self.offset)
        ].reset_index(drop=True)
    
    def load_embeddings(self):
        """Import patch embeddings."""
        if self.config.data.emb:
            emb = pd.read_csv(self.config.data.emb)
            emb = emb.loc[emb.cell_id.isin(self.csv.id)]
            self.csv = self.csv.set_index('id').loc[emb.cell_id].reset_index()
            self.emb = emb.iloc[:, 1:].to_numpy(dtype=np.float32)
        else:
            self.emb = np.zeros((len(self.csv), 0), dtype=np.float32)

    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        x = self.crop(idx)
        x = x if self.config.data.inmemory else self.transform(x)
        emb = self.emb[idx]
        lab = self.csv.at[idx, 'label']
        return torch.from_numpy(x), torch.from_numpy(emb), torch.tensor([lab])

    def crop(self, idx, window=None):
        """Crop thumbnail around a cell."""
        offset = (window // 2) if window else self.offset
        xcenter, ycenter = self.csv.at[idx, 'x'], self.csv.at[idx, 'y']
        xstart, xend = xcenter - offset, xcenter + offset
        ystart, yend = ycenter - offset, ycenter + offset
        return self.img[:, ystart:yend, xstart:xend]

    def transform(self, img):
        """Transform image or patch."""
        return img.astype('float32') / 255

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
            np.save('data/valid_idx.npy', self.valid_idx)
        self.train_idx = np.setdiff1d(np.arange(len(self.dataset)), self.valid_idx)

    def create_split(self):
        """Create balanced train/test split indices."""
        valid_ratio = 1 - self.config.train.train_ratio
        valid_idx = []
        for label in [0, 1]:
            class_idx = np.where(self.dataset.csv.label == label)[0]
            valid_size = int(len(class_idx) * valid_ratio)
            valid_idx.append(np.random.choice(class_idx, valid_size, replace=False))
        self.valid_idx = np.random.permutation(np.concatenate(valid_idx))
    
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
