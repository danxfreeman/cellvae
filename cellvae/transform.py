import numpy as np
import torch

class Vignette:

    def __init__(self, crop_size, alpha):
        x = np.linspace(-1, 1, crop_size, dtype=np.float32)
        xx, yy = np.meshgrid(x, x)
        r = np.sqrt(xx**2 + yy**2) / np.sqrt(2)
        mask = np.cos(0.5*np.pi * np.clip(r**alpha, 0, 1))
        mask[mask < 0] = 0
        self.mask = torch.tensor(mask)
    
    def __call__(self, x):
        return torch.as_tensor(x) * self.mask[None, :, :]

class IFTransform:

    def __init__(self, qmin=0, qmax=1):
        self.qmin = qmin
        self.qmax = qmax
    
    def __call__(self, x):
        x = np.log1p(x)
        xmin, xmax = np.quantile(x, q=(self.qmin, self.qmax), axis=(0, 2, 3), keepdims=True)
        x = (x - xmin) / (xmax - xmin)
        return x.clip(0, 1).astype(np.float32)
