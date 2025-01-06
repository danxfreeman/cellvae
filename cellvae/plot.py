import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

class Plot:

    def __init__(self, config, dataset=None):
        self.config = config
        self.dataset = dataset
    
    def _plot_cell(self, ax, id_, window=None, outline=False):
        window = window or self.config.preprocess.crop_size
        idx = self.dataset.csv.index[self.dataset.csv.id == id_][0]
        x = self.dataset.crop(idx, window=window)
        x = x.transpose(1, 2, 0)
        ax.imshow(x)
        ax.axis('off')
        if outline:
            circle = Circle(
                (window / 2, window / 2),
                radius = self.config.preprocess.crop_size / 2,
                edgecolor='white',
                facecolor='none',
                linewidth=3,
                linestyle=(0, (1, 1)),
                alpha=0.5
            )
            ax.add_patch(circle)
    
    def cell(self, id_, **kwargs):
        _, ax = plt.subplots()
        self._plot_cell(ax, id_=id_, **kwargs)
        plt.show()

    def grid(self, ids, cols=None, rows=None, **kwargs):
        N = len(ids)
        cols = cols or int(np.ceil(np.sqrt(N)))
        rows = rows or int(np.ceil(N / cols))
        _, axs = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axs = np.array(axs).flatten()
        for i, ax in enumerate(axs):
            if i < N:
                self._plot_cell(ax=ax, id_=ids[i], **kwargs)
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
