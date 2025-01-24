from typing import Any
from sklearn.metrics import r2_score

import torch
from torch import Tensor, tensor

from torchmetrics.metric import Metric
from torchmetrics import R2Score, MeanSquaredError
import numpy as np

def compute_r2(y_true, y_pred, aggr_mode='uniform_average'):
    if len(y_true.shape) <= 2:
        return r2_score(y_true, y_pred, multioutput=aggr_mode)
    
    if len(y_true.shape) == 3:
        return np.mean([
            r2_score(y_true[..., i], y_pred[..., i], multioutput=aggr_mode) for i in range(y_true.shape[-1])
        ])
    
    raise NotImplementedError(f'Cannot apply on y of {len(y_true.shape)} dims')

def _compute_corr_for_one_dim(y_true, y_pred):
    sigma_p = y_pred.std()
    if sigma_p == 0:
        return None
    
    sigma_g = y_true.std()
    mean_p = y_pred.mean()
    mean_g = y_true.mean()
    sigma_p += 1e-7
    sigma_g += 1e-7
    correlation = ((y_pred - mean_p) * (y_true - mean_g)).mean() / (sigma_p * sigma_g)
    return correlation

def compute_corr(y_true, y_pred):
    if (len(y_true.shape) == 2 and y_true.shape[-1] == 1) or (len(y_true.shape) == 1):
        return _compute_corr_for_one_dim(y_true, y_pred)

    if len(y_true.shape) == 3:
        return np.mean([
            compute_corr(y_true[i, ...], y_pred[i, ...]) for i in range(y_true.shape[0])
        ])

    if len(y_true.shape) == 2:
        if isinstance(y_pred, torch.Tensor):
            sigma_p = y_pred.std(1, correction=0)
        else:
            sigma_p = y_pred.std(1)
            
        if isinstance(y_true, torch.Tensor):
            sigma_g = y_true.std(1, correction=0)
        else:
            sigma_g = y_true.std(1)
        mean_p = y_pred.mean(1).reshape(-1, 1)
        mean_g = y_true.mean(1).reshape(-1, 1)
        index = (sigma_g != 0)
        sigma_p += 1e-7
        sigma_g += 1e-7
        correlation = ((y_pred - mean_p) * (y_true - mean_g)).mean(1) / (sigma_p * sigma_g)
        correlation = correlation[index].mean().item()
        return correlation
    
    raise NotImplementedError(f'Cannot apply on y of {len(y_true.shape)} dims')

class TrendAcc(Metric):
    """Metric for single step forcasting"""
    compute_by_all : bool = False
    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_trend_hit", default=tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor, xt: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        assert preds.shape == target.shape
        assert len(preds.shape) == 2
        batch_size = preds.shape[0]
        self.sum_trend_hit += ((preds - xt) * (target - xt) > 0).sum()
        self.total += batch_size * preds.shape[1]

    def compute(self) -> Tensor:
        """Computes mean squared error over state."""
        return self.sum_trend_hit / self.total

class R2(R2Score):
    """Metric for multi step forcasting"""
    compute_by_all : bool = False

    def __init__(self, num_outputs, adjusted: int = 0, multioutput: str = "uniform_average") -> None:
        super().__init__(num_outputs, adjusted, multioutput)

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets."""
        R2Score.update(self, preds, target)
    def compute(self) -> Tensor:
        return R2Score.compute(self)

class Corr(Metric):
    """correlation for multivariate timeseries, Corr compute correlation for every columns/nodes and output the averaged result"""
    compute_by_all : bool = True
    def __init__(self, save_on_gpu=False):
        super().__init__()
        self.save_on_gpu = save_on_gpu
        if save_on_gpu == True:
            self.add_state("y_pred", default=torch.Tensor(), dist_reduce_fx="cat")
            self.add_state("y_true", default=torch.Tensor(), dist_reduce_fx="cat")
        else:
            self.add_state("y_pred", default=[])
            self.add_state("y_true", default=[])


    def update(self, y_pred: torch.Tensor, y_true: torch.Tensor):           
        if self.save_on_gpu == True:
            self.y_pred = torch.cat([self.y_pred, y_pred], dim=0)
            self.y_true = torch.cat([self.y_true, y_true], dim=0)
        else:
            self.y_pred.append(y_pred.detach().cpu().numpy())
            self.y_true.append(y_true.detach().cpu().numpy())

    def compute(self):
        if self.save_on_gpu == True:
            return compute_corr(self.y_pred, self.y_true)
        else:
            return compute_corr(np.concatenate(self.y_pred, axis=0), np.concatenate(self.y_true, axis=0))

class RMSE(MeanSquaredError):
    def compute(self):
        return torch.sqrt(super().compute())