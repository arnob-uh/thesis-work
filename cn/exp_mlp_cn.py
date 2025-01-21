import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_utils_updated import time_series_epoch_trainer, time_series_epoch_evaluator
from lob_loader_updated import TimeSeriesDataset
from cn import CN
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, input_size, window_size, hidden_size=64, output_size=7, mode='adaptive_avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.0001):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size*window_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.cn = CN(num_features=7)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cn(x)
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.size(0), 7 * 10)
        x = self.model(x)
        return x

def run_experiment(window_size, batch_size=32, lr=0.001, epochs=10):
    data_dir = '../ETT-small'
    train_file = os.path.join(data_dir, 'ETTh1.csv')

    dataset = TimeSeriesDataset(file_path=train_file, window_size=window_size, train=True)

    torch.manual_seed(42)
    indices = torch.randperm(len(dataset))

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    input_size = 7
    model = MLP(input_size=input_size, window_size=window_size, hidden_size=64, output_size=7)
    model.cuda()

    for epoch in range(epochs):
        train_loss_mse, train_loss_mae, train_r2 = time_series_epoch_trainer(model, train_loader, lr=lr)
        val_loss_mse, val_loss_mae, val_r2 = time_series_epoch_evaluator(model, val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train MSE: {train_loss_mse:.4f}, MAE: {train_loss_mae:.4f}, R²: {train_r2:.4f}")
        print(f"Validation MSE: {val_loss_mse:.4f}, MAE: {val_loss_mae:.4f}, R²: {val_r2:.4f}")

        if (epoch + 1) % 5 == 0:
            test_loss_mse, test_loss_mae, test_r2 = time_series_epoch_evaluator(model, test_loader)
            print(f"Test MSE: {test_loss_mse:.4f}, MAE: {test_loss_mae:.4f}, R²: {test_r2:.4f}\n")


    test_loss_mse, test_loss_mae, test_r2 = time_series_epoch_evaluator(model, test_loader)
    print(f"Test MSE: {test_loss_mse:.4f}, MAE: {test_loss_mae:.4f}, R²: {test_r2:.4f}")

run_experiment(window_size=10, batch_size=32, lr=0.0001, epochs=10)
