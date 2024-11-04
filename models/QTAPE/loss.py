import torch
import torch.nn.functional as F

def rmse_loss(y_pred, y_true):
    return torch.sqrt(F.mse_loss(y_pred, y_true))