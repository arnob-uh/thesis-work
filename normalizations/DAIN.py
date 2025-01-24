import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DAIN(nn.Module):
    def __init__(self, mean_lr=0.00001, gate_lr=0.001, scale_lr=0.00001, feature_dim=96):
        super(DAIN, self).__init__()

        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        self.mean_layer = nn.Linear(feature_dim, feature_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(np.eye(feature_dim, feature_dim))

        self.scaling_layer = nn.Linear(feature_dim, feature_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(np.eye(feature_dim, feature_dim))

        self.gating_layer = nn.Linear(feature_dim, feature_dim)

        self.eps = 1e-8

    def loss(self, true):
        return 0

    def forward(self, x):
        # Step 1: Adaptive averaging
        avg = torch.mean(x, 2, keepdim=True)
        adaptive_avg = self.mean_layer(avg.squeeze(-1))
        adaptive_avg = adaptive_avg.unsqueeze(-1)
        x = x - adaptive_avg

        # Step 2: Adaptive scaling
        std = torch.mean(x ** 2, 2, keepdim=True)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std.squeeze(-1))
        adaptive_std = adaptive_std.unsqueeze(-1)
        adaptive_std[adaptive_std <= self.eps] = 1
        x = x / adaptive_std

        # Step 3: Adaptive gating
        avg = torch.mean(x, 2, keepdim=True)
        gate = torch.sigmoid(self.gating_layer(avg.squeeze(-1)))
        gate = gate.unsqueeze(-1)
        x = x * gate

        return x
