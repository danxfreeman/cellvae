# Import modules.
import os
import pandas as pd
import torch
import logging
from datetime import datetime
from cellvae import model
from torch import nn

# VAE Agent.
class CellAgent:
    def __init__(self, config, loader):
        self.config = config
        self.loader = loader

        # Initialize file paths used in the experiment.
        self.ckpt_file = os.path.join(self.config.input.output, 'checkpoint.pth.tar')
        self.loss_file = os.path.join(self.config.input.output, 'loss.csv')
        
        # Initialize model.
        torch.manual_seed(self.config.model.seed)
        self.model = model.CellVAE(self.config, mode='train')
        self.opt = torch.optim.Adam(self.model.parameters(), 
            lr=self.config.model.learning_rate)
        
        # Load or initialize weights.
        logging.info(f'Looking for checkpoint file at {self.ckpt_file}')
        try:
            self.load_checkpoint()
            logging.info(f'Checkpoint loaded at epoch {self.current_epoch}')
        except OSError:
            logging.info(f'No checkpoint file found. Applying random initialization.')
            self.current_epoch = 0
            self.init_weights()

        # Use GPU if available.
        if torch.cuda.is_available():
            logging.info(f'Running on {torch.cuda.device_count()} GPUs')
            self.model = nn.DataParallel(self.model)
            self.device = torch.device('cuda')
        else:
            logging.info(f'Running on {self.config.model.cpu_threads} CPUs')
            torch.set_num_interop_threads(self.config.model.cpu_threads)
            self.device = torch.device('cpu')
        self.model.to(self.device)

    # Initlize weights.
    def init_weights(self):
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    # Save checkpoint.
    def save_checkpoint(self):
        try:
            state = {
                'epoch': self.current_epoch,
                'model': self.model.modules.state_dict(),
                'optimizer': self.opt.modules.state_dict(),
            }
        except AttributeError:
            state = {
                'epoch': self.current_epoch,
                'model': self.model.state_dict(),
                'optimizer': self.opt.state_dict(),
            }
        torch.save(state, self.ckpt_file)

    # Load checkpoint.
    def load_checkpoint(self):
        checkpoint = torch.load(self.ckpt_file)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model'])
        self.opt.load_state_dict(checkpoint['optimizer'])

    # Log batch status.
    def update_log_batch(self):
        cur_epoch = self.current_epoch + 1
        tot_epoch = self.config.model.epochs
        cur_batch = self.current_batch
        tot_batch = len(self.loader.train_loader)
        cur_epoch = str(cur_epoch).zfill(len(str(tot_epoch)))
        cur_batch = str(cur_batch).zfill(len(str(tot_batch)))
        epoch = f'epoch {cur_epoch} of {tot_epoch}'
        batch = f'batch {cur_batch} of {tot_batch}'
        logging.info(f'Training {epoch}, {batch}')
    
    # Log epoch status.
    def update_log_epoch(self):
        epoch = self.current_epoch + 1
        epoch = str(epoch).zfill(len(str(self.config.model.epochs)))
        train_kdl = f'train_kld={float(self.train_kld_loss):.4f}'
        train_mse = f'train_mse={float(self.train_mse_loss):.4f}'
        valid_kdl = f'valid_kdl={float(self.test_kld_loss):.4f}'
        valid_mse = f'valid_mse={float(self.test_mse_loss):.4f}'
        logging.info(f'Finished epoch {epoch}: {train_kdl}, {train_mse}, {valid_kdl}, {valid_mse}')

    # Save loss to CSV file.
    def update_lossfile(self):
        log = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'epoch': self.current_epoch,
            'train_kdl_loss': float(self.train_kld_loss),
            'train_mse_loss': float(self.train_mse_loss),
            'train_tot_loss': float(self.train_total_loss),
            'valid_kdl_loss': float(self.test_kld_loss),
            'valid_mse_loss': float(self.test_mse_loss),
            'valid_tot_loss': float(self.test_total_loss)
        }
        df = pd.DataFrame([log])
        if os.path.isfile(self.loss_file):
            df.to_csv(self.loss_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.loss_file, mode='w', header=True, index=False)

    # Main operator.
    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            logging.info('You have entered CTRL-C. Wait to finalize.')
            self.save_checkpoint()
            self.update_lossfile()

    # Train model.
    def train(self):
        while self.current_epoch < self.config.model.epochs:
            self.train_one_epoch()
            self.validate()
            self.update_log_epoch()
            self.update_lossfile()
            self.save_checkpoint()
            self.current_epoch += 1
    
    # Train one epoch.
    def train_one_epoch(self):
        self.model.mode = 'train'
        self.model.train()
        self.current_batch = 0
        self.train_total_loss = 0
        self.train_mse_loss = 0
        self.train_kld_loss = 0
        for x in self.loader.train_loader:
            self.opt.zero_grad()
            x = x.to(self.device)
            x_hat, mu, log_var = self.model(x)
            total_loss, mse_loss, kld_loss = self.loss_function(x, x_hat, mu, log_var)
            self.train_total_loss += total_loss
            self.train_mse_loss += mse_loss
            self.train_kld_loss += kld_loss
            total_loss.backward()
            self.opt.step()
            self.current_batch += 1
            if self.current_batch % 100 == 0:
                self.update_log_batch()
        self.train_total_loss /= len(self.loader.train_loader)
        self.train_mse_loss /= len(self.loader.train_loader)
        self.train_kld_loss /= len(self.loader.train_loader)
    
    # Validate model.
    def validate(self):
        self.model.mode = 'test'
        self.model.eval()
        self.test_total_loss = 0
        self.test_mse_loss = 0
        self.test_kld_loss = 0
        if self.loader.valid_loader is None:
            return
        with torch.no_grad():
            for x in self.loader.valid_loader:
                x = x.to(self.device)
                x_hat, mu, log_var = self.model(x)
                total_loss, mse_loss, kld_loss = self.loss_function(x, x_hat, mu, log_var)
                self.test_total_loss += total_loss
                self.test_mse_loss += mse_loss
                self.test_kld_loss += kld_loss
        self.test_total_loss /= len(self.loader.valid_loader)
        self.test_mse_loss /= len(self.loader.valid_loader)
        self.test_kld_loss /= len(self.loader.valid_loader)

    # Reconstruct cells.
    def predict(self, cells):
        recon = torch.zeros([len(cells)] + list(cells[0].shape))
        self.model.mode = 'test'
        self.model.eval()
        with torch.no_grad():
            for i, x in enumerate(cells):
                x = x[None, ]
                x = x.to(self.device)
                x_hat = self.model(x)[0]
                recon[i, ] = x_hat
        return recon
        
    # Embed cells.
    def embed(self, cells):
        embedding = torch.zeros(len(cells), self.config.model.latent_dim)
        self.model.mode = 'test'
        self.model.eval()
        with torch.no_grad():
            for i, x in enumerate(cells):
                x = x[None, ]
                x = x.to(self.device)
                z = self.model.encoder(x)[0]
                embedding[i, ] = z
        return embedding
    
    # Define loss function.
    def loss_function(self, x, x_hat, mu, log_var):
        log_var = log_var.clip(min=-4, max=3)
        mse_loss = torch.mean(torch.abs(x - x_hat))
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        if self.config.model.kld_anneal:
            epoch = torch.tensor(self.current_epoch)
            kld_anneal = 1 / (1 + torch.exp(-1 * (epoch - 10)))
        else:
            kld_anneal = torch.tensor(1)
        kld_weighted = kld_loss * kld_anneal * self.config.model.kld_weight
        total_loss = mse_loss + kld_weighted
        return total_loss, mse_loss, kld_loss

# Adam vs. RMSProp?
