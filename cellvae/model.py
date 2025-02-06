import torch
import torch.nn as nn

class CellCNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.config.preprocess.crop_size, self.config.preprocess.crop_size)
            emb_size = self.config.data.emb_use * self.config.data.emb_size
            fc_input = self.conv_layers(dummy).numel() + emb_size
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input, 128),
            nn.ReLU(),
            nn.Dropout(p=self.config.model.dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, x, p):
        x = self.conv_layers(x)
        x = torch.hstack([x, p])
        x = self.fc_layers(x)
        return x
