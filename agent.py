# Import modules.
import os
import pandas as pd
import torch
import logging
from datetime import datetime
import model

# VAE Agent.
class CellAgent:
    def __init__(self, config, loader):
        self.config = config
        self.loader = loader

        # Initialize file paths used in the experiment.
        self.loss_file = os.path.join(self.config.input.output, 'loss.csv')
        self.ckpt_file = os.path.join(self.config.input.output, 'checkpoint.pth.tar')
        logging.info('****Initializing experiment****')

        # Use GPU if available.
        if torch.cuda.is_available():
            logging.info(f'Running on {torch.cuda.device_count()} GPUs')
            self.model = torch.nn.DataParallel(self.model)
            self.device = torch.device('cuda')
        else:
            logging.info(f'Running on {self.config.model.cpu_threads} CPUs')
            torch.set_num_interop_threads(self.config.model.cpu_threads)
            self.device = torch.device('cpu')
        
        # Initialize model.
        torch.manual_seed(self.config.model.seed)
        self.model = model.CellVAE(self.config, mode='train')
        self.opt = torch.optim.Adam(self.model.parameters(), 
            lr=self.config.model.learning_rate)
        self.model.to(self.device)
        self.load_checkpoint()
        
    # Update loss file.
    def update_lossfile(self):
        log = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'time': datetime.now().strftime('%H:%M:%S'),
            'epoch': self.current_epoch,
            'train_kdl_loss': self.train_kld_loss.detach().numpy(),
            'train_mse_loss': self.train_mse_loss.detach().numpy(),
            'train_tot_loss': self.train_total_loss.detach().numpy(),
            'valid_kdl_loss': self.test_kld_loss.detach().numpy(),
            'valid_mse_loss': self.test_mse_loss.detach().numpy(),
            'valid_tot_loss': self.test_total_loss.detach().numpy()
        }
        df = pd.DataFrame([log])
        if os.path.isfile(self.loss_file):
            df.to_csv(self.loss_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.loss_file, mode='w', header=True, index=False)

    # Update log file.
    def update_logfile(self):
        cur_epoch = self.current_epoch
        tot_epoch = self.config.model.epochs
        cur_batch = self.current_batch
        tot_batch = len(self.loader.train_loader)
        cur_epoch = str(cur_epoch).zfill(len(str(tot_epoch)))
        cur_batch = str(cur_batch).zfill(len(str(tot_batch)))
        epoch = f'epoch {cur_epoch} of {tot_epoch}'
        batch = f'batch {cur_batch} of {tot_batch}'
        loss = self.train_total_loss / self.current_batch
        loss = f'loss={loss:.4f}'
        logging.info(f'Training {epoch}, {batch}: {loss}')

    # Save checkpoint.
    def save_checkpoint(self):
        logging.info(f'Saving checkpoint to {self.ckpt_file}')
        state = {
            'epoch': self.current_epoch,
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
        }
        torch.save(state, self.ckpt_file)

    # Load checkpoint.
    def load_checkpoint(self):
        try:
            logging.info(f'Looking for checkpoint file at {self.ckpt_file}')
            checkpoint = torch.load(self.ckpt_file)
            self.current_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['model'])
            self.opt.load_state_dict(checkpoint['optimizer'])
            logging.info(f'Checkpoint loaded at epoch {self.current_epoch}')
        except OSError:
            logging.info(f'No checkpoint file found')

    # Main operator.
    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            logging.info('You have entered CTRL-C. Wait to finalize.')
            self.save_checkpoint()
            self.update_lossfile()
            self.update_logfile()

    # Train model.
    def train(self):
        for epoch in range(self.config.model.epochs):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.validate()
            self.update_lossfile()
            if (epoch > 0) & (epoch % 10 == 0):
                self.save_checkpoint()
    
    # Train one epoch.
    def train_one_epoch(self):
        self.model.mode = 'train'
        self.train_total_loss = 0
        self.train_mse_loss = 0
        self.train_kld_loss = 0
        for i, x in enumerate(self.loader.train_loader):
            self.current_batch = i
            self.opt.zero_grad()
            x = x.to(self.device)
            x_hat, mu, log_var = self.model(x)
            total_loss, mse_loss, kld_loss = self.loss_function(x, x_hat, mu, log_var)
            self.train_total_loss += total_loss
            self.train_mse_loss += mse_loss
            self.train_kld_loss += kld_loss
            total_loss.backward()
            self.opt.step()
            if i % 100 == 0:
                self.update_logfile()
        self.train_total_loss /= len(self.loader.train_loader)
        self.train_mse_loss /= len(self.loader.train_loader)
        self.train_kld_loss /= len(self.loader.train_loader)
    
    # Validate model.
    def validate(self):
        self.model.mode = 'test'
        self.test_total_loss = 0
        self.test_mse_loss = 0
        self.test_kld_loss = 0
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

    # Define loss function.
    def loss_function(self, x, x_hat, mu, log_var):
        log_var = log_var.clip(min=-4, max=3)
        mse_loss = torch.mean((x - x_hat)**2)
        kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        kld_loss *= self.config.model.kld_scaler
        total_loss = mse_loss + kld_loss
        return total_loss, mse_loss, kld_loss

# Adam vs. RMSProp?

