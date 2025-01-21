import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_utils_new import time_series_epoch_trainer, time_series_epoch_evaluator
from data_provider.data_factory import data_provider
from normalizations.dain import DAIN_Layer

class MLP(nn.Module):
    def __init__(self, input_size, window_size, hidden_size=32, output_size=120, mode='adaptive_avg', mean_lr=0.00001, gate_lr=0.001, scale_lr=0.0001):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size * window_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_size, output_size)
        )

        self.dain = DAIN_Layer(mode=mode, mean_lr=mean_lr, gate_lr=gate_lr, scale_lr=scale_lr, input_dim=12)

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        x = x.transpose(1, 2)
        x = self.dain(x)
        x = x.transpose(1, 2)

        x = x.contiguous().view(batch_size, -1)
        
        x = self.model(x)
        x = x.view(batch_size, seq_len, num_features)

        return x

def run_experiment(args, window_size, batch_size=32, lr=0.001, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, train_loader = data_provider(args, flag='train')
    val_set, val_loader = data_provider(args, flag='val')
    test_set, test_loader = data_provider(args, flag='test')

    input_size = 12
    model = MLP(input_size=input_size, window_size=window_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss_mse, train_loss_mae, train_r2 = time_series_epoch_trainer(model, train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)
        val_loss_mse, val_loss_mae, val_r2 = time_series_epoch_evaluator(model, val_loader, loss_fn=loss_fn, device=device)

        print(f"Training - MSE: {train_loss_mse:.4f}, MAE: {train_loss_mae:.4f}, R²: {train_r2:.4f}")
        print(f"Validation - MSE: {val_loss_mse:.4f}, MAE: {val_loss_mae:.4f}, R²: {val_r2:.4f}")

    test_loss_mse, test_loss_mae, test_r2 = time_series_epoch_evaluator(model, test_loader, loss_fn=loss_fn, device=device)
    print(f"Test - MSE: {test_loss_mse:.4f}, MAE: {test_loss_mae:.4f}, R²: {test_r2:.4f}")

if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.root_path = './data/'
            self.data_path = 'data_all.csv'
            self.data = 'telco'
            self.seq_len = 10
            self.label_len = 5
            self.pred_len = 5
            self.features = 'M'
            self.target = 'value'
            self.embed = 'timeF'
            self.freq = 't'
            self.batch_size = 32

    args = Args()
    run_experiment(args, window_size=10, batch_size=32, lr=0.001, epochs=10)
