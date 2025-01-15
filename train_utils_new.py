import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import r2_score

def time_series_epoch_trainer(model, loader, optimizer, loss_fn, device):
    model.train()

    model_optimizer = optimizer
    train_loss_mse, train_loss_mae, counter = 0, 0, 0
    train_r2_scores = []

    for (inputs, targets, _, _) in loader:
        model_optimizer.zero_grad()
        inputs, targets = inputs.to(device).float(), targets.to(device).float()

        predictions = model(inputs)
        
        loss_mse = loss_fn(predictions, targets)
        loss_mae = torch.nn.functional.l1_loss(predictions, targets)

        loss_mse.backward()
        model_optimizer.step()

        predictions_reshaped = predictions.view(predictions.size(0), -1).cpu().detach().numpy()
        targets_reshaped = targets.view(targets.size(0), -1).cpu().detach().numpy()

        r2 = r2_score(targets_reshaped, predictions_reshaped)
        train_r2_scores.append(r2)

        train_loss_mse += loss_mse.item()
        train_loss_mae += loss_mae.item()
        counter += 1

    avg_mse_loss = train_loss_mse / counter
    avg_mae_loss = train_loss_mae / counter
    avg_r2_score = np.mean(train_r2_scores)
    
    return avg_mse_loss, avg_mae_loss, avg_r2_score

def time_series_epoch_evaluator(model, loader, loss_fn, device):
    model.eval()
    
    val_loss_mse, val_loss_mae, counter = 0, 0, 0
    val_r2_scores = []

    with torch.no_grad():
        for (inputs, targets, _, _) in loader:
            inputs, targets = inputs.to(device).float(), targets.to(device).float()

            predictions = model(inputs)

            loss_mse = loss_fn(predictions, targets).item()
            loss_mae = torch.nn.functional.l1_loss(predictions, targets).item()

            predictions_reshaped = predictions.view(predictions.size(0), -1).cpu().detach().numpy()
            targets_reshaped = targets.view(targets.size(0), -1).cpu().detach().numpy()

            r2 = r2_score(targets_reshaped, predictions_reshaped)
            val_r2_scores.append(r2)

            val_loss_mse += loss_mse
            val_loss_mae += loss_mae
            counter += 1

    avg_val_mse = val_loss_mse / counter
    avg_val_mae = val_loss_mae / counter
    avg_val_r2 = np.mean(val_r2_scores)
    
    return avg_val_mse, avg_val_mae, avg_val_r2

