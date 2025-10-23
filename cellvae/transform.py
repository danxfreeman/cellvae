import numpy as np
import torch

class Vignette:

    def __init__(self, alpha=0, crop_size=32):
        x = np.linspace(-1, 1, crop_size, dtype=np.float32)
        r = np.sqrt(x[:, None]**2 + x[None, :]**2) / np.sqrt(2)
        mask = np.cos(0.5*np.pi * r**alpha)
        mask = np.nan_to_num(mask.clip(0, 1))
        self.mask = torch.tensor(mask)
    
    def __call__(self, x):
        return x * self.mask[None, :, :]

class ClipTransform:

    def __init__(self, qmin=0, qmax=1):
        self.qmin = qmin
        self.qmax = qmax
    
    def __call__(self, x):
        xmin, xmax = np.quantile(x, q=(self.qmin, self.qmax), axis=(0, 2, 3), keepdims=True)
        x = (x - xmin) / (xmax - xmin)
        return np.clip(x, 0, 1).astype(np.float32)

class LogTransform:

    def __call__(self, x):
        return np.log1p(x)
