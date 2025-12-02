import torch

# ---------------------------
# Helper to flatten tensors
# ---------------------------
def _flatten(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    return pred, target


# ---------------------------
# Binarizer
# ---------------------------
def _binarize(pred, threshold=0.5):
    return (pred > threshold).float()


# ---------------------------
# DICE SCORE
# ---------------------------
def dice_score(pred, target, eps=1e-7):
    pred = _binarize(pred)
    pred, target = _flatten(pred, target)

    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)


# ---------------------------
# IOU / JACCARD
# ---------------------------
def iou_score(pred, target, eps=1e-7):
    pred = _binarize(pred)
    pred, target = _flatten(pred, target)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection + eps) / (union + eps)


# ---------------------------
# PRECISION
# ---------------------------
def precision_score(pred, target, eps=1e-7):
    pred = _binarize(pred)
    pred, target = _flatten(pred, target)

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()

    return (tp + eps) / (tp + fp + eps)


# ---------------------------
# RECALL
