import os
from.dataset import Dataset, Freq, TimeSeriesDataset
import pandas as pd
import numpy as np

class TELCO(TimeSeriesDataset):
    name:str= 'TELCO'
    num_features: int = 12
    freq : Freq = 't'
    length : int  = 61056
    windows : int = 384
    
    def download(self):
        pass

    def _load(self) -> np.ndarray:
        self.file_path = os.path.join(self.dir, 'telco.csv')
        self.df = pd.read_csv(self.file_path,parse_dates=[0])
        self.dates = pd.DataFrame({'date':self.df.time})
        self.data = self.df.iloc[:, 1:].to_numpy()
        return self.data
    