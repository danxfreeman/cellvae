import json
import logging

import torch
import pandas as pd

from datetime import datetime
from tqdm import tqdm

from cellvae.model import CellCNN

class CellAgent:

    def __init__(self, config, loader):
        self.config = config
        self.loader = loader
        torch.manual_seed(self.config.model.seed)
        self.model = CellCNN(self.config)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.model.learning_rate)
        self.loss = torch.nn.BCELoss()
        self.current_epoch = 0
        self.load_checkpoint()
        logging.info(f'Config\n{json.dumps(self.config, indent=4)}')
        logging.info(f'Model\n{self.model}')

    def load_checkpoint(self):
        """Load checkpoint if available."""
        logging.info('Looking for checkpoint file...')
        try:
            checkpoint = torch.load('data/checkpoint.pth.tar')
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])
            self.opt.load_state_dict(checkpoint['optimizer'])
            logging.info(f'Checkpoint loaded at epoch {self.current_epoch}')
        except FileNotFoundError:
            logging.info('No checkpoint found. Creating new model.')
    
    def save_checkpoint(self):
        """Save checkpoint."""
        state = {
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict()
        }
        torch.save(state, 'data/checkpoint.pth.tar')

    def run(self):
        """Main operator."""
        try:
            self.train()
        except KeyboardInterrupt:
            logging.info('Process interrupted. Exiting.')
        finally:
            logging.info('Processing complete.')

    def train(self):
        """Train model."""
        while self.current_epoch < self.config.train.num_epochs:
            logging.info(f'Training epoch {self.current_epoch} of {self.config.train.num_epochs}.')
            self.train_one_epoch()
            self.validate()
            self.save_checkpoint()
            self.save_loss()
            self.current_epoch += 1
    
    def train_one_epoch(self):
        """Train one epoch."""
        self.train_loss = 0
        self.model.train()
        for idx, (x, y) in enumerate(self.loader.train_loader):
            self.opt.zero_grad()
            y_pred = self.model(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            self.opt.step()
            self.train_loss += loss.item()
            if idx % 100 == 0:
                logging.info(f'Training batch {idx} of {len(self.loader.train_loader)}')
    
    def validate(self):
        """Measure validation loss."""
        self.valid_loss = 0
        self.model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.loader.valid_loader):
                y_pred = self.model(x)
                loss = self.loss(y_pred, y)
                self.valid_loss += loss.item()
                if idx % 100 == 0:
                    logging.info(f'Validating batch {idx} of {len(self.loader.valid_loader)}')
    
    def predict(self):
        """Classify validation set."""
        self.model.eval()
        with torch.no_grad():
            for x, _ in tqdm(self.loader.valid_loader, desc='Predicting'):
                for y_pred in self.model(x):
                    yield y_pred.item()

    def save_loss(self):
        """Save loss to CSV."""
        log = pd.DataFrame([{
            'time': str(datetime.now()),
            'epoch': self.current_epoch,
            'train_loss': self.train_loss / len(self.loader.train_set),
            'valid_loss': self.valid_loss / len(self.loader.valid_set)
        }])
        save_header = self.current_epoch == 0
        log.to_csv('data/loss.csv', mode='a', index=False, header=save_header)
