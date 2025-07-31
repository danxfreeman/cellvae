import numpy as np

class Vignette:

    def __init__(self, config):
        self.vignette = config.preprocess.vignette
        self.crop_size = config.preprocess.crop_size
        x = np.linspace(-self.vignette, self.vignette, self.crop_size)
        x = np.exp(-x ** 2 / 2)
        self.mask = x[:, None] * x
    
    def __call__(self, x):
        return x * self.mask[None, :, :]

def transform_channels(thumbnails, min_quant, max_quant):
    """Transform IF data of shape N C X Y"""
    thumbnails = np.log1p(thumbnails.astype(np.float32))
    qmin, qmax = np.quantile(thumbnails, q=(min_quant, max_quant), axis=(0, 2, 3), keepdims=True)
    thumbnails = (thumbnails - qmin) / (qmax - qmin)
    return np.clip(thumbnails, 0, 1)

