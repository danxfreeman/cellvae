import torch
import torch.nn as nn

class CellCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, stride=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, stride=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config.preprocess.crop_size, config.preprocess.crop_size)
            self.fc_input = self.conv_layers(dummy).numel()
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input, 128),
            nn.ReLU(),
            nn.Dropout(config.model.dropout_prob),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
