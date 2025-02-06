import json
import logging

import torch
import torch.nn.functional as F
import pandas as pd
import wandb

from datetime import datetime

from cellvae.model import CellVAE

class CellAgent:

    def __init__(self, config, loader):
        self.config = config
        self.loader = loader
        torch.manual_seed(self.config.model.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CellVAE(self.config).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.model.learning_rate)
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
        self.train_mse_loss = 0
        self.train_kld_loss = 0
        self.train_sum_loss = 0
        self.model.train()
        for idx, x in enumerate(self.loader.train_loader):
            self.opt.zero_grad()
            x = x.to(self.device)
            x_hat, mu, logvar = self.model(x)
            mse_loss, kld_loss, sum_loss = self.loss(x, x_hat, mu, logvar)
            sum_loss.backward()
            self.opt.step()
            self.train_mse_loss += mse_loss.item()
            self.train_kld_loss += kld_loss.item()
            self.train_sum_loss += sum_loss.item()
            if idx % 100 == 0:
                logging.info(f'Training batch {idx} of {len(self.loader.train_loader)}.')

    def validate(self):
        """Evaluate model."""
        self.valid_mse_loss = 0
        self.valid_kld_loss = 0
        self.valid_sum_loss = 0
        self.model.eval()
        with torch.no_grad():
            for idx, x in enumerate(self.loader.valid_loader):
                x = x.to(self.device)
                x_hat, mu, logvar = self.model(x)
                mse_loss, kld_loss, sum_loss = self.loss(x, x_hat, mu, logvar)
                self.valid_mse_loss += mse_loss.item()
                self.valid_kld_loss += kld_loss.item()
                self.valid_sum_loss += sum_loss.item()
                if idx % 100 == 0:
                    logging.info(f'Validating batch {idx} of {len(self.loader.valid_loader)}.')

    def loss(self, x, x_hat, mu, logvar):
        """Calculate loss."""
        mse_loss = F.mse_loss(x_hat, x)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        sum_loss = mse_loss + (kld_loss * self.config.model.beta)
        return mse_loss, kld_loss, sum_loss

    def embed(self):
        """Embed validation set."""
        self.model.eval()
        with torch.no_grad():
            for i, x in enumerate(self.loader.valid_loader):
                logging.info(f'Embedding batch {i} of {len(self.loader.valid_loader)}.')    
                x_hat_batch, z_batch, _ = self.model(x)
                for x_hat, z in zip(x_hat_batch, z_batch):
                    yield x_hat, z

    def save_loss(self):
        """Save loss to CSV."""
        log = pd.DataFrame([{
            'time': str(datetime.now()),
            'epoch': self.current_epoch,
            'train_bce_loss': self.train_bce_loss / len(self.loader.train_set),
            'train_kld_loss': self.train_kld_loss / len(self.loader.train_set),
            'train_sum_loss': self.train_sum_loss / len(self.loader.train_set),
            'valid_bce_loss': self.valid_bce_loss / len(self.loader.valid_set),
            'valid_kld_loss': self.valid_kld_loss / len(self.loader.valid_set),
            'valid_sum_loss': self.valid_sum_loss / len(self.loader.valid_set),
        }])
        save_header = self.current_epoch == 0
        log.to_csv('data/loss.csv', mode='a', index=False, header=save_header)
        wandb.log(log.iloc[:, 2:].to_dict(), step=self.current_epoch)
