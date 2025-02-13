import torch.nn as nn

class ZScore(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, batch_x):
        return self.normalize(batch_x)
    
    def loss(self, x1):
        return 0
    
    def normalize(self, batch_x):
        mean = batch_x.mean(dim=0, keepdim=True)
        std = batch_x.std(dim=0, keepdim=True, unbiased=False)
        return (batch_x - mean) / (std + 1e-8)
    