import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import torch.optim as optim
from sklearn.metrics import r2_score

def time_series_epoch_trainer(model, train_loader, lr=0.0001, optimizer=optim.Adam):
    model.train()

    model_optimizer = optimizer(model.parameters(), lr=lr)
    total_loss_mse = 0.0
    total_loss_mae = 0.0
    total_r2 = 0.0

    device = next(model.parameters()).device

    for batch in train_loader:
        inputs, targets = batch
        
        inputs = inputs.to(device)
        targets = targets.to(device)

        model_optimizer.zero_grad()

        predictions = model(inputs)

        #RevIN
        targets = targets.unsqueeze(1).repeat(1, predictions.size(1), 1)
        
        #Dain, CN
        #targets = targets.view_as(predictions)    

        loss_mse = torch.nn.functional.mse_loss(predictions, targets)
        loss_mae = torch.nn.functional.l1_loss(predictions, targets)

        loss_mse.backward()
        model_optimizer.step()
        
        predictions_mean = predictions.mean(dim=1)
        targets_mean = targets.mean(dim=1)
        
        r2 = r2_score(targets_mean.cpu().detach().numpy(), predictions_mean.cpu().detach().numpy())
        
        total_loss_mse += loss_mse.item()
        total_loss_mae += loss_mae.item()
        total_r2 += r2
    
    return total_loss_mse / len(train_loader), total_loss_mae / len(train_loader), total_r2 / len(train_loader)

def time_series_epoch_evaluator(model, val_loader):
    model.eval()
    total_loss_mse = 0.0
    total_loss_mae = 0.0
    total_r2 = 0.0
    
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)

            #RevIN
            targets = targets.unsqueeze(1).repeat(1, predictions.size(1), 1)

            #Dain, CN
            # targets = targets.view_as(predictions)

            loss_mse = torch.nn.functional.mse_loss(predictions, targets)
            loss_mae = torch.nn.functional.l1_loss(predictions, targets)

            predictions_mean = predictions.mean(dim=1)
            targets_mean = targets.mean(dim=1)

            r2 = r2_score(targets_mean.cpu().detach().numpy(), predictions_mean.cpu().detach().numpy())

            total_loss_mse += loss_mse.item()
            total_loss_mae += loss_mae.item()
            total_r2 += r2
    
    return total_loss_mse / len(val_loader), total_loss_mae / len(val_loader), total_r2 / len(val_loader)
