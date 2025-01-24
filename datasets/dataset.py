import numpy as np
import pandas as pd
from typing import Optional, Sequence
import torch.utils.data
import os
from abc import abstractmethod
from enum import Enum, unique

@unique
class Freq(str, Enum):
    seconds = "s"
    minutes = "t"
    hours = "h"
    days = "d"
    months = "m"
    years = "y"

class Dataset(torch.utils.data.Dataset):
    name: str
    num_features: int
    length: int
    freq: Freq

    def __init__(self, root: str):
        super().__init__()

        self.data: np.ndarray
        self.df: pd.DataFrame

    def download(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

StoreTypes = np.ndarray

class TimeSeriesDataset(Dataset):
    def __init__(self, root: str='./data'):
        super().__init__(root)
        self.root = root
        self.dir = os.path.join(root, self.name)
        os.makedirs(self.dir, exist_ok=True)
        
        self.download()
        self._process()
        self._load()
        
        self.dates: Optional[pd.DataFrame ]
        
    @abstractmethod
    def download(self):
        raise NotImplementedError

    def _process(self) :
        pass
    
    @abstractmethod
    def _load(self) -> StoreTypes:
        raise NotImplementedError

class TimeSeriesStaticGraphDataset(TimeSeriesDataset):
    adj : np.ndarray 
    def _load_static_graph(self):
        raise NotImplementedError()

class TimeseriesSubset(torch.utils.data.Subset):
    def __init__(self, dataset: TimeSeriesDataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices
        self.data = self.dataset.data[indices]
        self.df = self.dataset.df.iloc[indices]
        self.dates = self.dataset.dates.iloc[indices]
        self.num_features = dataset.num_features
        self.name = dataset.name
        self.length = len(self.indices)
        self.freq = dataset.freq
