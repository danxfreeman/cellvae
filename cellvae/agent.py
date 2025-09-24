import os
import logging
from datetime import datetime

import torch
import torch.nn.functional as F
import pandas as pd

from cellvae.model import CellVAE

class CellAgent:

    def __init__(self, config, loader=None, in_channels=None, outdir='results'):
        self.config = config
        self.loader = loader
        os.makedirs(outdir, exist_ok=True)
        self.weights_path = f'{outdir}/checkpoint.pth.tar'
        self.loss_path = f'{outdir}/loss.csv'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_channels = in_channels or self.loader.dataset[0].shape[0]
        self.model = CellVAE(self.config, in_channels=in_channels).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.model.learning_rate)
        self.current_epoch = 1
        self.load_checkpoint()

    def load_checkpoint(self):
        """Load checkpoint if available."""
        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])
            self.opt.load_state_dict(checkpoint['optimizer'])
            logging.info(f'Checkpoint loaded at epoch {self.current_epoch}.')
        except FileNotFoundError:
            logging.info('No checkpoint found. Creating new model.')
    
    def save_checkpoint(self):
        """Save checkpoint."""
        state = {
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict()
        }
        torch.save(state, self.weights_path)

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
        while self.current_epoch <= self.config.train.num_epochs:
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
        mse_loss = F.mse_loss(x_hat, x, reduction='sum') / x.size(0)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        sum_loss = mse_loss + (kld_loss * self.config.model.beta)
        return mse_loss, kld_loss, sum_loss

    def _infer(self, x, fn, batch_size=10):
        """Process dataset in batches."""
        self.model.eval()
        with torch.no_grad():
            for i, (x_batch,) in enumerate(x):
                x_batch = x_batch.to(self.device)
                if i % batch_size == 0:
                    logging.info(f'Processing batch {i} of {len(x)}.')
                yield fn(x_batch).detach().cpu()
    
    def encode(self, x):
        """Encode thumbnails."""
        yield from self._infer(x, lambda x: self.model.encoder(x)[0])

    def decode(self, x):
        """Reconstruct thumbnails."""
        yield from self._infer(x, lambda x: self.model(x)[0])

    def save_loss(self):
        """Save loss to CSV."""
        len_train = len(self.loader.train_loader) or 1
        len_valid = len(self.loader.valid_loader) or 1
        log = pd.DataFrame([{
            'time': str(datetime.now()),
            'epoch': self.current_epoch,
            'train_mse_loss': self.train_mse_loss / len_train,
            'train_kld_loss': self.train_kld_loss / len_train,
            'train_sum_loss': self.train_sum_loss / len_train,
            'valid_mse_loss': self.valid_mse_loss / len_valid,
            'valid_kld_loss': self.valid_kld_loss / len_valid,
            'valid_sum_loss': self.valid_sum_loss / len_valid,
        }])
        if self.current_epoch == 1:
            log.to_csv(self.loss_path, mode='w', header=True, index=False)
        else:
            log.to_csv(self.loss_path, mode='a', header=False, index=False)
