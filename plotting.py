# Import modules.
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import cycle, islice

# Create class with plotting functions.
class Plot():
    def __init__(self, config, dataset):
        self.config = config
        self.dataset = dataset

    # Plot thubmnails.
    def plot_thumbnails(self, cell_index, channel_index, channel_name=None,
            channel_color=None, channel_boost=None, title=None, 
            filename=None):
        self.cell_index = cell_index
        self.channel_index = channel_index
        self.channel_name = channel_name
        self.channel_color = channel_color
        self.channel_boost = channel_boost
        if not channel_name:
            self.channel_name = [self.config.input.channel_name[i] for i in self.channel_index]
        if not channel_color:
            colors = mcolors.TABLEAU_COLORS
            self.channel_color = list(islice(cycle(colors), len(channel_index)))
        if not channel_boost:
            self.channel_boost = [1] * len(channel_index)
        self.plot_all_cells()
        self.fig.suptitle(title)
        if filename:
            self.fig.savefig(filename)
        else:
            return self.fig
    
    # Plot cell by channel grid.
    def plot_all_cells(self):
        self.fig = plt.figure(layout='tight')
        self.axs = self.fig.subplots(nrows=len(self.cell_index), ncols=len(self.channel_index))
        for cell_i, cell_idx in enumerate(self.cell_index):
            self.cell_i = cell_i
            self.cell_idx = cell_idx
            self.cell_lab = str(cell_idx).zfill(len(str(np.max(self.channel_index))))
            self.cell_name = f'Cell{self.cell_lab}'
            for ch_i, ch_idx in enumerate(self.channel_index):
                self.ch_i = ch_i
                self.ch_idx = ch_idx
                self.ch_name = self.channel_name[ch_i]
                self.ch_color = self.channel_color[ch_i]
                self.ch_boost = self.channel_boost[ch_i]
                self.xlab = self.ch_name if cell_i == 0 else None
                self.ylab = self.cell_name if ch_i == 0 else None
                self.plot_one_cell()
    
    # Plot thumbnail for one cell and channel.
    def plot_one_cell(self):
        x = self.dataset[self.cell_idx][self.ch_idx, :, :]
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

    # Plot channels.
    def plot_channels(self, channel_index=None, channel_name=None, filename=None):
        self.channel_index = channel_index
        self.channel_name = channel_name
        if channel_index is None:
            self.channel_index = list(range(len(self.config.input.channel_name)))
        if channel_name is None:
            self.channel_name = [self.config.input.channel_name[i] for i in self.channel_index]
        self.plot_channel_grid()
        if filename:
            self.fig.savefig(filename)
        else:
            return self.fig

    # Plot channel grid.
    def plot_channel_grid(self):
        self.fig = plt.figure(figsize=(12, 14), layout='tight') # temp
        nrows = int(np.ceil(np.sqrt(len(self.channel_index))))
        ncols = int(np.ceil(len(self.channel_index) / nrows))
        self.axs = self.fig.subplots(nrows=nrows, ncols=ncols)
        for row in range(nrows):
            for col in range(ncols):
                ch_idx = row * ncols + col
                if ch_idx < len(self.channel_index):
                    self.row = row
                    self.col = col
                    self.ch_idx = ch_idx
                    self.ch_name = self.channel_name[ch_idx]
                    self.plot_one_channel()

    # Plot histogram for one channel.
    def plot_one_channel(self):
        dat = [x[self.ch_idx, :, :] for x in self.dataset]
        dat = [x.flatten() for x in dat]
        dat = np.concatenate(dat, axis=0)
        ax = self.axs[self.row, self.col]
        ax.hist(dat, bins=40, range=(0, 1), color='#619CFF', alpha=0.5, ec='black')
        ax.set_title(self.ch_name)
        ax.set_box_aspect(1)
        ax.set_yticks([])
        ax.margins(x=0)
