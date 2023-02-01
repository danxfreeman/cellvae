# Import modules.
import os
import numpy as np
import tifffile as tiff
import logging

# Preprocess image.
def process_img(img, csv, config):
    n_channels = img.shape[0]
    img = img.astype(np.float16)
    for c in range(n_channels):
        logging.info(f'Preprocessing channel {c}')
        img[c, :, :] = process_one_channel(img[c, :, :], config)
    print(f'Saving thumbnails to {config.input.thumbnails}')
    create_thumbnails(img, csv, config)

# Preprocess one channel.
def process_one_channel(channel, config):
    min_cutoff = np.quantile(channel[channel > 0], config.preprocess.min_quantile)
    max_cutoff = np.quantile(channel, config.preprocess.max_quantile)
    channel = np.clip(channel, min_cutoff, max_cutoff)
    channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
    if config.preprocess.log_transform:
        channel = np.log1p(channel)
    return channel

# Crop a thumbnail around each cell.
def create_thumbnails(img, csv, config):
    crop_size = config.preprocess.crop_size
    xmax = img.shape[1] - crop_size
    ymax = img.shape[2] - crop_size
    expand = crop_size // 2
    for i in range(len(csv)):
        xcenter = csv[i, 1]
        ycenter = csv[i, 0] # temp
        xstart = xcenter - expand
        xstart = np.clip(xstart, 0, xmax)
        xend = xstart + crop_size
        ystart = ycenter - expand
        ystart = np.clip(ystart, 0, ymax)
        yend = ystart + crop_size
        thumbnail = img[:, xstart:xend, ystart:yend]
        file_ = os.path.join(config.input.thumbnails, f'cell_{i}.tif')
        channel_names = config.input.channel_name
        metadata = {'axes': 'CXY', 'Channel': {'Name': channel_names}}
        tiff.imwrite(file_, data=thumbnail, metadata=metadata)
        if i % 1000 == 0:
            cur_cell = str(i).zfill(len(str(len(csv))))
            logging.info(f'Cropping cell {cur_cell} of {len(csv)}')
