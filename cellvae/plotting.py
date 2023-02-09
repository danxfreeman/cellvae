# Import modules.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import Subset

# Class with plotting functions.
class Plot():
    def __init__(self, config, dataset, cell_index=None):
        self.config = config
        self.dataset = dataset
        self.cell_index = cell_index
        if self.cell_index is None:
            self.cell_index = 10
        if type(self.cell_index) is int:
            self.cell_index = np.random.choice(len(self.dataset), self.cell_index, replace=False)
            self.cell_index = np.sort(self.cell_index)
        self.dataset = Subset(self.dataset, self.cell_index)
        
    # Plot cell thubmnails.
    def plot_thumbnails(self, channel_index=None, channel_name=None, 
        channel_color=None, channel_boost=None):
        self.channel_index = channel_index
        self.channel_name = channel_name
        self.channel_color = channel_color
        self.channel_boost = channel_boost
        if self.channel_index is None:
            self.channel_index = np.arange(len(self.config.input.channel_name))
        if self.channel_name is None:
            self.channel_name = np.array(self.config.input.channel_name)[self.channel_index]
        if self.channel_color is None:
            colors = list(mcolors.TABLEAU_COLORS.values())
            self.channel_color = np.resize(colors, len(self.channel_index))
        if self.channel_boost is None:
            self.channel_boost = np.ones(len(self.channel_index))
        self.__plot_all_cells()
        return self.fig
    
    # Plot cell by channel grid.
    def __plot_all_cells(self):
        self.fig = plt.figure(layout='tight')
        nrows = len(self.cell_index)
        ncols = len(self.channel_index)
        self.fig.set_size_inches(1.2*ncols, 1.2*nrows)
        self.axs = self.fig.subplots(nrows=nrows, ncols=ncols)
        for cell_i, cell_idx in enumerate(self.cell_index):
            self.cell_i = cell_i
            self.cell_idx = cell_idx
            self.cell_name = str(cell_idx).zfill(len(str(np.max(self.cell_index))))
            self.cell_name = f'Cell_{self.cell_name}'
            for ch_i, ch_idx in enumerate(self.channel_index):
                self.ch_i = ch_i
                self.ch_idx = ch_idx
                self.ch_name = self.channel_name[ch_i]
                self.ch_color = self.channel_color[ch_i]
                self.ch_boost = self.channel_boost[ch_i]
                self.xlab = self.ch_name if cell_i == 0 else None
                self.ylab = self.cell_name if ch_i == 0 else None
                self.__plot_one_cell()
    
    # Plot one cell.
    def __plot_one_cell(self):
        x = self.dataset[self.cell_i][self.ch_idx, :, :]
        x = np.stack([x, x, x], axis=-1)
        x = x * mcolors.to_rgb(self.ch_color)
        x = x * self.ch_boost
        x = np.clip(x, 0, 1)
        ax = self.axs[self.cell_i, self.ch_i]
        ax.imshow(x)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(self.xlab)
        ax.set_ylabel(self.ylab)
        ax.xaxis.set_label_position('top')

    # Plot channel histograms.
    def plot_channels(self, channel_index=None, channel_name=None):
        self.channel_index = channel_index
        self.channel_name = channel_name
        if self.channel_index is None:
            self.channel_index = np.arange(len(self.config.input.channel_name))
        if self.channel_name is None:
            self.channel_name = np.array(self.config.input.channel_name)[self.channel_index]
        self.__plot_all_channels()
        return self.fig

    # Plot channel grid.
    def __plot_all_channels(self):
        self.fig = plt.figure(layout='tight')
        nrows = int(np.ceil(np.sqrt(len(self.channel_index))))
        ncols = int(np.ceil(len(self.channel_index) / nrows))
        self.fig.set_size_inches(2*ncols, 2*nrows)
        self.axs = self.fig.subplots(nrows=nrows, ncols=ncols)
        for row in range(nrows):
            for col in range(ncols):
                self.row = row
                self.col = col
                self.ch_idx = row * ncols + col
                if self.ch_idx < len(self.channel_index):
                    self.ch_name = self.channel_name[self.ch_idx]
                    self.__plot_one_channel()
                else:
                    self.axs[self.row, self.col].set_axis_off()

    # Plot one channel.
    def __plot_one_channel(self):
        x = [y[self.ch_idx, :, :] for y in self.dataset]
        x = np.stack(x).flatten()
        ax = self.axs[self.row, self.col]
        ax.hist(x, range=(0, 1), bins=40, color='#619CFF', alpha=0.5, ec='black')
        ax.set_title(self.ch_name)
        ax.set_box_aspect(1)
        ax.set_yticks([])
        ax.margins(x=0)
    
    # Plot reconstruction and KDL loss.
    def plot_loss(self):
        path = os.path.join(self.config.input.output, 'loss.csv')
        loss = pd.read_csv(path)
        fig = plt.figure()
        axs = fig.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        axs[0].plot(loss.epoch, loss.train_kdl_loss, color='orange', label='KDL')
        axs[0].plot(loss.epoch, loss.train_mse_loss, color='blue', label='MSE')
        axs[0].plot(loss.epoch, loss.train_tot_loss, color='lime', label='Total')
        axs[0].set_title('Training')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[1].plot(loss.epoch, loss.valid_kdl_loss, color='orange')
        axs[1].plot(loss.epoch, loss.valid_mse_loss, color='blue')
        axs[1].plot(loss.epoch, loss.valid_tot_loss, color='lime')
        axs[1].set_title('Validation')
        axs[1].set_xlabel('Epoch')
        fig.legend(loc = 'lower center', ncol=3)
        fig.subplots_adjust(bottom=0.2)
        return fig
