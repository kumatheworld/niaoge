import torch
from sklearn.metrics import f1_score

def to_ndarray(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else x

def binarize(pred, threshold=0.5):
    return pred > threshold

def mean_f1_score(pred, label, threshold=0.5):
    y_true = to_ndarray(label)
    y_pred = to_ndarray(binarize(pred, threshold))
    return f1_score(y_true, y_pred, average='samples', zero_division=1)
