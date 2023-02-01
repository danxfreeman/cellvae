# Import modules.
import numpy as np
import torch
from torch import nn
from torchvision import transforms

# Define autoencoder.
class CellVAE(nn.Module):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config
        self.mode = mode
        self.input_size = self.config.preprocess.crop_size
        self.input_pad = self.conv_padding()
        self.config.model.conv_dim = [self.config.input.n_channels] + self.config.model.conv_dim
        self.config.model.final_dim = self.config.model.conv_dim[-1]
        self.config.model.final_height = self.conv_output_size()
        self.config.model.fc_in = self.config.model.final_dim * self.config.model.final_height ** 2
        self.encoder = CellEncoder(self.config)
        self.decoder = CellDecoder(self.config)
    
    # Pad input to the nearest power of two.
    def conv_padding(self):
        new_size = 2 ** int(np.ceil(np.log2(self.input_size)))
        input_pad = (new_size - self.input_size) // 2
        return input_pad
    
    # Calculate output size from convolutional layers.
    def conv_output_size(self):
        size = 2 ** int(np.ceil(np.log2(self.input_size)))
        layers = len(self.config.model.conv_dim) - 1
        kernel = self.config.model.kernel_size
        stride = self.config.model.stride
        padding = 1
        for _ in range(layers):
            size = int(((size - kernel + (2 * padding)) / stride) + 1)
        return size

    # Reparameterize.
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    # Forward pass.
    def forward(self, x):
        x = transforms.Pad(self.input_pad)(x)
        mu, log_var = self.encoder(x)
        if self.mode == 'train':
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        x_hat = self.decoder(z)
        x_hat = transforms.CenterCrop(self.input_size)(x_hat)
        return x_hat, mu, log_var

# Define encoder.
class CellEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_dim = self.config.model.conv_dim
        self.fc_in = self.config.model.fc_in
        self.latent_dim = self.config.model.latent_dim
        self.encoder = []

        # Build convolutional layer.
        for k in range(len(self.conv_dim) - 1):
            in_dim = self.conv_dim[k]
            out_dim = self.conv_dim[k + 1]
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU()
                )
            )
        
        # Build fully connected layer.
        self.encoder.append(nn.Flatten())
        self.encoder = nn.Sequential(*self.encoder)
        self.fc_mu = nn.Linear(self.fc_in, self.latent_dim)
        self.fc_var = nn.Linear(self.fc_in, self.latent_dim)

    # Forward pass.
    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

# Define decoder.
class CellDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_dim = self.config.model.conv_dim
        self.fc_in = self.config.model.fc_in
        self.final_dim = self.config.model.conv_dim[-1]
        self.final_height = self.config.model.final_height
        self.latent_dim = self.config.model.latent_dim
        self.decoder = []
        
        # Build fully connected layer.
        self.decoder.append(
            nn.Sequential(
                nn.Linear(self.latent_dim, self.fc_in),
                nn.BatchNorm1d(self.fc_in),
                nn.LeakyReLU()
            )
        )
        
        # Build convolutional layer.
        input_dim = [self.final_dim, self.final_height, self.final_height]
        self.decoder.append(nn.Unflatten(1, input_dim))
        for k in range(len(self.conv_dim) - 1):
            in_dim = self.conv_dim[::-1][k]
            out_dim = self.conv_dim[::-1][k + 1]
            activation = nn.LeakyReLU if k < len(self.conv_dim) - 2 else nn.Sigmoid
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, 
                        padding=1, output_padding=1),
                    nn.BatchNorm2d(out_dim),
                    activation() # use sigmoid activation in the final layer
                )
            )
        self.decoder = nn.Sequential(*self.decoder)
    
    # Forward pass.
    def forward(self, z):
        x = self.decoder(z)
        return x

# ReLU vs. LeakyReLU?