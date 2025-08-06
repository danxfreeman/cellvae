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

class LogTransform:

    def __init__(self, config):
        self.qmin = config.preprocess.min_quant
        self.qmax = config.preprocess.max_quant
    
    def __call__(self, x):
        xmin, xmax = np.quantile(x, q=(self.qmin, self.qmax), axis=(0, 2, 3), keepdims=True)
        x = (x - xmin) / (xmax - xmin)
        return np.clip(x, 0, 1).astype(np.float32)
