import torch

def expectile_loss(diff, tau=0.9):
    """
    Asymmetric loss to emphasize the upper envelope of safety values.
    diff = Target - Prediction
    """
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * (diff ** 2)).mean()

