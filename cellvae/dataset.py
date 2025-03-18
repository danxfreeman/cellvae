import numpy as np

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import ToTensor

class CellDataset(Dataset):

    def __init__(self, dirname='data'):
        self.thumbnails = np.load(f'{dirname}/thumbnails.npy')
        self.transform = ToTensor()

    def __len__(self):
        return len(self.thumbnails)
    
    def __getitem__(self, idx):
        x = self.thumbnails[idx]
        return self.transform(x)

class CellLoader:

    def __init__(self, config):
        self.config = config
        self.dataset = CellDataset()
        self.train_idx = np.load('data/train_idx.npy')
        self.valid_idx = np.load('data/valid_idx.npy')
        self.split_dataset()
    
    def split_dataset(self):
        """Split dataset into train and test sets."""
        self.train_set = Subset(self.dataset, self.train_idx)
        self.valid_set = Subset(self.dataset, self.valid_idx)
        self.train_loader = self.load_dataset(self.train_set)
        self.valid_loader = self.load_dataset(self.valid_set)
    
    def load_dataset(self, dataset):
        """Load dataset into DataLoader."""
        if len(dataset) == 0:
            return []
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.config.train.batch_size,
            num_workers=self.config.train.num_workers
        )
