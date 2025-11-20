import torch
import torch.nn.functional as F


def dice_score(pred, target, smooth=1e-6):
    """
    pred: logits (B,1,H,W)
    target: masks (B,1,H,W)
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2 * intersection + smooth) / (union + smooth)

    return dice.mean().item()


def iou_score(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()
