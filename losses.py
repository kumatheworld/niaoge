import torch
from torch.nn.functional import binary_cross_entropy

def BCELoss(pred, label):
    return binary_cross_entropy(pred, label)

def SoftF1Loss(pred, label):
    numerator = torch.sum(torch.abs(pred - label), dim=-1)
    denominator = torch.sum(pred + label, dim=-1)
    return torch.mean(numerator / denominator)
