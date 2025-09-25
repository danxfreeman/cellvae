import numpy as np
import torch

class Vignette:

    def __init__(self, config):
        x = np.linspace(-1, 1, config.preprocess.crop_size, dtype=np.float32)
        xx, yy = np.meshgrid(x, x)
        r = np.sqrt(xx**2 + yy**2) / np.sqrt(2)
        alpha, beta = config.preprocess.alpha, config.preprocess.beta
        mask = np.cos(0.5*np.pi * np.clip(r**alpha, 0, 1))**beta
        mask[mask < 0] = 0
        self.mask = torch.tensor(mask)
    
    def __call__(self, x):
        return x * self.mask[None, :, :]

class LogTransform:

    def __init__(self, config):
        self.qmin = config.preprocess.min_quant
        self.qmax = config.preprocess.max_quant
    
    def __call__(self, x):
        x = np.log1p(x)
        xmin, xmax = np.quantile(x, q=(self.qmin, self.qmax), axis=(0, 2, 3), keepdims=True)
        x = (x - xmin) / (xmax - xmin)
        return np.clip(x, 0, 1).astype(np.float32)

