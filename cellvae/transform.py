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
