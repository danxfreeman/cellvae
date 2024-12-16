import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

class Plot:

    def __init__(self, config):
        self.config = config
    
    def grid(self, dataset, idx):
        cols = int(np.ceil(np.sqrt(len(idx))))
        rows = int(np.ceil(len(idx) / cols))
        _, axs = plt.subplots(rows, cols, figsize=(2*rows,2*cols))
        axs = np.array(axs).flatten()
        for i, ax in enumerate(axs):
            if i < len(idx):
                x = dataset[idx[i]][0]
                x = x.numpy().transpose(1, 2, 0)
                ax.imshow(x)
                ax.axis('off')
            else:
                ax.set_visible(False)
        plt.tight_layout()
        plt.show()
    
    def loss(self):
        loss = pd.read_csv('data/loss.csv')
        plt.figure(figsize=(10, 5))
        plt.plot(loss['epoch'], loss['train_loss'], label='Train Loss', marker='o')
        plt.plot(loss['epoch'], loss['valid_loss'], label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
