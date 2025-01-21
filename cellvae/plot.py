import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.patches import Circle

class Plot:

    def __init__(self, config, dataset=None):
        self.config = config
        self.dataset = dataset
        self.loss_df = None
    
    def cell(self, id_, **kwargs):
        _, ax = plt.subplots()
        self._plot_cell(ax, id_=id_, **kwargs)
        plt.show()

    def gallery(self, ids, cols=None, rows=None, **kwargs):
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
    
    def _plot_cell(self, ax, id_, window=None, outline=False):
        window = window or self.config.preprocess.crop_size
        try:
            idx = self.dataset.csv.index[self.dataset.csv.id == id_][0]
        except IndexError:
            raise ValueError(f'Cell ID {id_} does not exist.')
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

    def loss(self, metric='loss'):
        if self.loss_df is None:
            self.loss_df = pd.read_csv('data/loss.csv')
            self.loss_df.epoch += 1
        if metric == 'loss':
            self._plot_curve(self.loss_df['train_loss'], self.loss_df['valid_loss'])
            plt.ylim(0, None)
            plt.ylabel('Loss')
        else:
            self._plot_curve(self.loss_df['train_auc'], self.loss_df['valid_auc'])
            plt.ylim(0.5, 1)
            plt.ylabel('AUC')
        plt.show()

    def _plot_curve(self, y_train, y_valid):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_df['epoch'], y_train, label='Train', marker='o')
        plt.plot(self.loss_df['epoch'], y_valid, label='Test', marker='o')
        plt.xlabel('Epoch')
        plt.gca().set_xlim(1, self.config.train.num_epochs)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.legend()
