import json
import logging

import torch
import pandas as pd

from torchmetrics.classification import BinaryAUROC
from datetime import datetime

from cellvae.model import CellCNN

class CellAgent:

    def __init__(self, config, loader, outdir='results/', weights_path=None):
        self.config = config
        self.loader = loader
        self.outdir = outdir
        self.weights_path = weights_path
        self.loss_path = f'{self.outdir}/loss.csv'
        torch.manual_seed(self.config.model.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CellCNN(self.config).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.config.model.learning_rate, weight_decay=self.config.model.weight_decay)
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.config.model.pos_weight))
        self.current_epoch = 1
        self.load_checkpoint()
        logging.info(json.dumps(self.config, indent=4))
        logging.info(self.model)

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
        self.train_loss = 0
        self.train_auc = BinaryAUROC()
        self.model.train()
        for idx, (x, y) in enumerate(self.loader.train_loader):
            self.opt.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            self.opt.step()
            self.train_loss += loss.item()
            self.train_auc.update(torch.sigmoid(y_pred), y)
            if idx % 100 == 0:
                logging.info(f'Training batch {idx} of {len(self.loader.train_loader)}.')

    def validate(self):
        """Evaluate model."""
        self.valid_loss = 0
        self.valid_auc = BinaryAUROC()
        self.model.eval()
        with torch.no_grad():
            for idx, (x, y) in enumerate(self.loader.valid_loader):
                x, y = x.to(self.device), y.to(self.device)
                y_pred = self.model(x)
                loss = self.loss(y_pred, y)
                self.valid_loss += loss.item()
                self.valid_auc.update(torch.sigmoid(y_pred), y)
                if idx % 100 == 0:
                    logging.info(f'Validating batch {idx} of {len(self.loader.valid_loader)}.')

    def infer(self, loader):
        """Generate predictions."""
        self.model.eval()
        with torch.no_grad():
            for idx, (x, _) in enumerate(loader):
                if idx % 100 == 0:
                    logging.info(f'Evaluating batch {idx} of {len(loader)}.')
                x = x.to(self.device)
                y_pred_batch = torch.sigmoid(self.model(x))
                for y_pred in y_pred_batch:
                    yield y_pred.item()

    def save_loss(self):
        """Save loss to CSV."""
        len_train = len(self.loader.train_loader)
        len_valid = len(self.loader.valid_loader)
        log = pd.DataFrame([{
            'time': str(datetime.now()),
            'epoch': self.current_epoch,
            'train_loss': self.train_loss / len_train,
            'train_auc': self.train_auc.compute().item(),
            'valid_loss': (self.valid_loss / len_valid) if len_valid else 0,
            'valid_auc': self.valid_auc.compute().item() if len_valid else 0
        }])
        save_header = self.current_epoch == 1
        log.to_csv(self.loss_path, mode='a', index=False, header=save_header)
