
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, window_size, train=True):
        self.window_size = window_size
        self.train = train

        self.data = pd.read_csv(file_path)

        self.data.pop('date')
        self.X = self.data.values[:-1]
        self.y = self.data.values

        num_features = self.X.shape[1]

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.window_size]
        # X_seq = X_seq.values.flatten()
        y_seq = self.y[idx + self.window_size]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)
