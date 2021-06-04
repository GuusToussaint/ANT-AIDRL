import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset
import os


class SARCOSDataset(Dataset):

    def __init__(self, root, train, transform=None) -> None:
        
        self.transform = transform

        data_file = 'sarcos_inv' if train else 'sarcos_inv_test'
        data_location = os.path.join(root, data_file + '.mat')

        # collect the raw data from the data files
        data = loadmat(data_location)[data_file].astype(np.float32)

        self.X, self.Y = data[:, :21], data[:, 21:]


    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        current_X = self.X[index]
        current_Y = self.Y[index]

        # TODO: maybe we should also transform the Y??
        if self.transform:
            current_X = self.transform(current_X)

        return current_X, current_Y