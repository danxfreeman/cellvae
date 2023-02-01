# Import modules.
import os
import pandas as pd
import tifffile as tiff
import logging
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import preprocess

# Create Dataset.
class CellDataset(Dataset):
    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform
        
        # Import channel labels.
        markers = pd.read_csv(self.config.input.markers)
        self.config.input.n_channels = len(markers)
        self.config.input.channel_name = list(markers.marker_name)
        self.config.input.channel_number = list(markers.channel_number)
        if not os.path.exists(self.config.input.output):
            os.makedirs(self.config.input.output)

        # Create log.
        self.logr_file = os.path.join(config.input.output, 'log.txt')
        logging.basicConfig(filename=self.logr_file, level=logging.INFO)
        logging.info('****Initializing dataset****')
        logging.info(f'Markers: {os.path.abspath(config.input.markers)}')
        logging.info(f'Image: {os.path.abspath(config.input.img)}')
        logging.info(f'CSV: {os.path.abspath(config.input.csv)}')
        logging.info(f'Output: {os.path.abspath(config.input.output)}')

        # Create thumbnails.
        self.config.input.thumbnails = os.path.join(self.config.input.output, 'thumbnails')
        logging.info(f'Checking for thumbnails at {self.config.input.thumbnails}')
        if not os.path.exists(self.config.input.thumbnails):
            logging.info(f'Creating thumbnails')
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
        self.train_set, self.valid_set = self.split_dataset(self.dataset)
        self.train_loader = self.load_dataset(self.train_set)
        self.valid_loader = self.load_dataset(self.valid_set)
    
    # Split dataset into training and validation sets.
    def split_dataset(self, dataset):
        train_size = int(self.config.loader.train_size * len(dataset))
        valid_size = len(dataset) - train_size
        train_set, valid_set = random_split(dataset, [train_size, valid_size])
        return train_set, valid_set
    
    # Load dataset.
    def load_dataset(self, dataset):
        return DataLoader(dataset,
            batch_size=self.config.loader.batch_size,
            shuffle=self.config.loader.shuffle,
            num_workers=self.config.loader.workers)
