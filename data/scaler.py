import torch
from torch import Tensor
from typing import Generic, TypeVar, Union

import pandas as pd
import numpy as np
import torch

StoreType = TypeVar(
    "StoreType", bound=Union[pd.DataFrame, np.ndarray, torch.Tensor]
)

class Scaler(Generic[StoreType]):
    def fit(self, data: StoreType) -> None:
        raise NotImplementedError()

    def transform(self, data: StoreType) -> StoreType:
        raise NotImplementedError()

    def inverse_transform(self, data: StoreType) -> StoreType:
        raise NotImplementedError()


class MaxAbsScaler(Scaler[StoreType]):
    def __init__(self) -> None:
        self.scale = None

    def fit(self, data: StoreType):
        if isinstance(data, np.ndarray):
            self.scale = np.max(np.abs(data), axis=0)
        elif isinstance(data, Tensor):
            self.scale = data.abs().max(axis=0).values
        else:
            raise ValueError(f"not supported type : {type(data)}")

    def transform(self, data) -> StoreType:
        return data / self.scale

    def inverse_transform(self, data: StoreType) -> StoreType:
        if isinstance(data, np.ndarray):
            return data * self.scale
        elif isinstance(data, Tensor):
            return data * torch.tensor(self.scale, device=data.device)
        else:
            raise ValueError(f"not supported type : {type(data)}")

class MinMaxScaler(Scaler):
    pass

class StandarScaler(Scaler):
    def __init__(self, device="cpu") -> None:
        self.mean = None
        self.std = None
        self.device = device
        

    def fit(self, data: StoreType):
        if isinstance(data, np.ndarray):
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
        elif isinstance(data, Tensor):
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0)
        else:
            raise ValueError(f"not supported type : {type(data)}")

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: StoreType) -> StoreType:
        if isinstance(data, np.ndarray):
            return data * self.std + self.mean
        elif isinstance(data, Tensor):
            return data * torch.tensor(self.std, device=data.device) + torch.tensor(
                self.mean, device=data.device
            )
        else:
            raise ValueError(f"not supported type : {type(data)}")

class NoScaler(Scaler):
    def __init__(self, device="cpu") -> None:
        self.device = device

    def fit(self, data: StoreType):
        pass

    def transform(self, data):
        return data

    def inverse_transform(self, data: StoreType) -> StoreType:
        return data
