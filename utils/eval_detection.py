# utils/eval_detection.py
import torch
import numpy as np

def iou_boxes(boxA, boxB):
    # box: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2]-boxA[0])) * max(0, (boxA[3]-boxA[1]))
    boxBArea = max(0, (boxB[2]-boxB[0])) * max(0, (boxB[3]-boxB[1]))
    denom = boxAArea + boxBArea - interArea
    if denom == 0:
        return 0.0
    return interArea / denom

def evaluate_simple(model, loader, device, iou_thresh=0.5, score_thresh=0.5):
    """
    Returns (precision, recall, f1) averaged across dataset.
    Note: not a COCO evaluator. Good for quick checks.
    """
    model.eval()
    tps = 0
    fps = 0
    fns = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs_t = [img.to(device) for img in imgs]
            outputs = model(imgs_t)
            for out, tgt in zip(outputs, targets):
                # convert tensors to numpy
                pred_boxes = out.get("boxes", torch.empty((0,4))).cpu().numpy()
                scores = out.get("scores", torch.empty((0,))).cpu().numpy()
                pred_labels = out.get("labels", torch.empty((0,))).cpu().numpy()

                # filter by score_thresh
                keep = scores >= score_thresh
                pred_boxes = pred_boxes[keep]
                # truth
                gt_boxes = tgt["boxes"].cpu().numpy() if "boxes" in tgt else np.zeros((0,4))
                matched_gt = np.zeros(len(gt_boxes), dtype=bool)

                # match preds to gts greedily
                for pb in pred_boxes:
                    found = False
                    best_iou = 0.0; best_j = -1
                    for j, gb in enumerate(gt_boxes):
                        i = iou_boxes(pb, gb)
                        if i > best_iou and not matched_gt[j]:
                            best_iou = i; best_j = j
                    if best_iou >= iou_thresh:
                        tps += 1
                        matched_gt[best_j] = True
                    else:
                        fps += 1
                # any unmatched gts are false negatives
                fns += (~matched_gt).sum()

    precision = tps / (tps + fps) if (tps + fps) > 0 else 0.0
    recall = tps / (tps + fns) if (tps + fns) > 0 else 0.0
    f1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
    return precision, recall, f1
