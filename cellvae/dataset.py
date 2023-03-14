# Import modules.
import os
import json
import logging
import numpy as np
import pandas as pd
import tifffile as tiff
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from cellvae import preprocess

# Create Dataset.
class CellDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform

        # Initialize logger.
        if not os.path.exists(config.input.output):
            os.makedirs(config.input.output)
        config_file = os.path.join(config.input.output, 'config.json')
        with open(config_file, 'w') as fp:
            json.dump(config, fp, indent=4)
        logr_file = os.path.join(config.input.output, 'log.txt')
        logging.basicConfig(filename=logr_file, level=logging.INFO)
        logging.info('****Initializing experiment****')
        
        # Create thumbnails.
        self.config.input.thumbnails = os.path.join(self.config.input.output, 'thumbnails/')
        logging.info(f'Looking for thumbnails at {self.config.input.thumbnails}')
        if not os.path.exists(self.config.input.thumbnails):
            logging.info(f'No thumbnails found. Creating thumbnails.')
            os.makedirs(self.config.input.thumbnails)
            csv = pd.read_csv(self.config.input.csv, usecols=self.config.input.csv_cols)
            csv = csv.to_numpy(dtype='int')
            img = tiff.imread(self.config.input.img, key=self.config.input.channel_number)
            preprocess.process_img(img, csv, self.config)
        else:
            logging.info('Thumbnails already exist')
    
    def __len__(self):
        files = os.listdir(self.config.input.thumbnails)
        return len(files)
    
    def __getitem__(self, idx):
        file_ = os.path.join(self.config.input.thumbnails, f'cell_{idx}.tif')
        thumbnail = tiff.imread(file_)
        if self.transform:
            thumbnail = self.transform(thumbnail)
        return torch.Tensor(thumbnail)

# Create DataLoader.
class CellLoader:
    def __init__(self, config):
        self.config = config
        self.dataset = CellDataset(self.config)
        self.split_dataset()
        self.train_set = Subset(self.dataset, self.train_idx)
        self.valid_set = Subset(self.dataset, self.valid_idx)
        self.train_loader = self.load_dataset(self.train_set)
        self.valid_loader = self.load_dataset(self.valid_set)
    
    # Split dataset into training and validation sets.
    def split_dataset(self):
        self.idx_file = os.path.join(self.config.input.output, 'train_idx.npy')
        logging.info(f'Looking for train indices at {self.idx_file}')
        if os.path.exists(self.idx_file):
            logging.info('Train indices loaded')
            self.load_split()
        else:
            logging.info('No train indices found. Applying random split.')
            self.random_split()
            np.save(self.idx_file, self.train_idx)

    # Load existing split.
    def load_split(self):
        self.train_idx = np.load(self.idx_file)
        self.valid_idx = np.setdiff1d(np.arange(len(self.dataset)), self.train_idx)

    # Apply random split.
    def random_split(self):
        train_size = int(self.config.loader.train_size * len(self.dataset))
        idx = np.arange(len(self.dataset))
        self.train_idx = np.random.choice(idx, train_size, replace=False)
        self.valid_idx = np.setdiff1d(idx, self.train_idx)
    
    # Load dataset.
    def load_dataset(self, dataset):
        if len(dataset) == 0:
            return None
        return DataLoader(dataset,
            batch_size=self.config.loader.batch_size,
            shuffle=self.config.loader.shuffle,
            num_workers=self.config.loader.workers)
