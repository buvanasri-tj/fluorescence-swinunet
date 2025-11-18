import torch

def dice_score(pred, target, eps=1e-7):
    """
    Computes Dice coefficient for binary segmentation.
    pred: raw logits from model (B,1,H,W)
    target: ground truth mask (B,1,H,W)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2 * intersection + eps) / (union + eps)
    return dice.item()
