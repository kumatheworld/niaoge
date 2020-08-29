import torch
from sklearn.metrics import f1_score
from statistics import mean

def binarize(pred, threshold=0.5):
    return pred > threshold

def mean_f1_score(y_true, y_pred):
    true_pred_pairs = torch.stack([y_true, y_pred], 1)
    true_pred_pairs_np = true_pred_pairs.cpu().detach().numpy()
    f1_scores = [f1_score(t, p) for t, p in true_pred_pairs_np]
    return mean(f1_scores)
