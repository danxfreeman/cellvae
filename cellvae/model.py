import torch
import torch.nn as nn

class CellVAE(nn.Module):

    def __init__(self, config, in_channels):
        super().__init__()
        self.config = config
        self.encoder = CellEncoder(config, in_channels=in_channels)
        self.decoder = CellDecoder(config, in_channels=in_channels, fc_dim=self.encoder.fc_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

class CellEncoder(nn.Module):
    def __init__(self, config, in_channels):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, stride=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, config.preprocess.crop_size, config.preprocess.crop_size)
            self.fc_dim = self.conv_layers(dummy).numel()
        self.fc_mu = nn.Linear(self.fc_dim, config.model.latent_dim)
        self.fc_logvar = nn.Linear(self.fc_dim, config.model.latent_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class CellDecoder(nn.Module):
    def __init__(self, config, in_channels, fc_dim):
        super().__init__()
        self.conv_dim = int((fc_dim // 128) ** 0.5)
        self.fc_dec = nn.Sequential(
            nn.Linear(config.model.latent_dim, fc_dim),
            nn.ReLU()
        )
        self.conv_layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=in_channels, stride=2, kernel_size=3, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc_dec(x)
        x = x.view(-1, 128, self.conv_dim, self.conv_dim)
        return self.conv_layers(x)
