import torch
from torch.nn import BCELoss

class SoftF1Loss_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, label):
        inv_avg = 2 / torch.sum(pred + label, dim=-1, keepdim=True)
        prod = torch.sum(pred * label, dim=-1, keepdim=True)
        f1 = inv_avg * prod
        grad_pred = inv_avg * (f1 - label)
        ctx.save_for_backward(grad_pred)
        return 1 - f1.mean()

    @staticmethod
    def backward(ctx, grad_output):
        grad_pred, = ctx.saved_tensors
        return grad_pred * grad_output, None

def SoftF1Loss():
    return SoftF1Loss_.apply
