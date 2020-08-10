import os
import torch
import numpy as np
from models import *
from torch.utils.data import Dataset, DataLoader

class ArrayDataset(Dataset):
    def __init__(self, datasets):
        super(ArrayDataset, self).__init__()
        self._length = len(datasets[0])
        for i, data in enumerate(datasets):
            assert len(data) == self._length, \
                "All arrays must have the same length; \
                array[0] has length %d while array[%d] has length %d." \
                % (self._length, i+1, len(data))
        self.datasets = datasets

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        return tuple(torch.from_numpy(data[idx]).float() \
                     for data in self.datasets)

class FakeDataset(Dataset):
    def __init__(self, n_fake=10000):
        super(FakeDataset, self).__init__()
        self.n_fake = n_fake
    
    def __len__(self):
        return self.n_fake
    
    def __getitem__(self, idx):
        return (0, 0)