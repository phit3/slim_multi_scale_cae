import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import transform

class SnapshotData(Dataset):
    @property
    def batch_size(self):
        if 'data_params' in self.config:
            if 'batch_size' in self.config['data_params']:
                return self.config['data_params']['batch_size']
        return 128

    @property
    def data_fname(self):
        if 'data_params' in self.config:
            if 'data_fname' in self.config['data_params']:
                return self.config['data_params']['data_fname']
        return 'D1'

    @property
    def width(self):
        if 'data_params' in self.config:
            if 'width' in self.config['data_params']:
                return self.config['data_params']['width']
        return 64

    @property
    def height(self):
        if 'data_params' in self.config:
            if 'height' in self.config['data_params']:
                return self.config['data_params']['height']
        return 64

    def __len__(self):
        return self.samples

    def __init__(self, data, config, subset, data_min=None, data_max=None, samples=None):
        self.config = config
        self.subset = subset
        self.data = data

        self.samples = samples if samples is not None else self.data.shape[0]

        self.data_min = data_min if data_min is not None else self.data.min()
        self.data_max = data_max if data_max is not None else self.data.max()
 
    def augment(self, x):
        max_h_margin = x.shape[1] - self.height
        max_w_margin = x.shape[2] - self.width
        h_shift = np.random.randint(0, max_h_margin + 1)
        w_shift = np.random.randint(0, max_w_margin + 1)
        x = x[:, h_shift:h_shift + self.height, w_shift:w_shift + self.width]
        return x

    def fix_augment(self, x):
        h_margin = (x.shape[1] - self.height) // 2
        w_margin = (x.shape[2] - self.width) // 2
        x = x[:, h_margin:h_margin + self.height, w_margin:w_margin + self.width]
        return x

    def __getitem__(self, idx):
        if idx >= self.data.shape[0]:
            idx = np.random.randint(0, self.data.shape[0])
        x = torch.from_numpy(self.data[idx, :, :, :])

        x = (x - self.data_min) / (self.data_max - self.data_min)

        if self.subset == 'train':
            x = self.augment(x)
        else:
            x = self.fix_augment(x)
        return x, x.clone()

