import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T
import csv

# Ensure repo root is in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.swinunet import SwinUNet
from datasets.dataset_seg import SegmentationDataset

# ---------------------------------------------------
# METRIC FUNCTIONS
# ---------------------------------------------------
def compute_metrics(pred, gt):
    """
    pred, gt are numpy arrays with values 0 or 1
    """

    TP = np.logical_and(pred == 1, gt == 1).sum()
    FP = np.logical_and(pred == 1, gt == 0).sum()
    FN = np.logical_and(pred == 0, gt == 1).sum()
    TN = np.logical_and(pred == 0, gt == 0).sum()

    # Avoid division errors
    eps = 1e-7

    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    dice = 2 * TP / (2 * TP + FP + FN + eps)

    return precision, recall, f1, accuracy, iou, dice


# ---------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------
def evaluate(model, test_loader, device, out_csv):

    rows = []
    metrics_sum = np.zeros(6)  # precision, recall, f1, accuracy, iou, dice

    with torch.no_grad():
        for idx, (img, mask) in enumerate(test_loader):

            img = img.to(device)
            gt = mask.squeeze().cpu().numpy()  # ground truth 0/1

            pred = torch.sigmoid(model(img)).cpu().numpy()[0, 0]
            pred_bin = (pred > 0.5).astype(np.uint8)

            precision, recall, f1, accuracy, iou, dice = compute_metrics(pred_bin, gt)

            metrics_sum += np.array([precision, recall, f1, accuracy, iou, dice])

            rows.append([idx, precision, recall, f1, accuracy, iou, dice])

    # Averages
    mean_metrics = metrics_sum / len(test_loader)

    # SAVE CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "precision", "recall", "f1", "accuracy", "iou", "dice"])
        writer.writerows(rows)
        writer.writerow(["MEAN"] + list(mean_metrics))

    print("\n[RESULTS]")
    print(f"Precision: {mean_metrics[0]:.4f}")
    print(f"Recall:    {mean_metrics[1]:.4f}")
    print(f"F1 Score:  {mean_metrics[2]:.4f}")
    print(f"Accuracy:  {mean_metrics[3]:.4f}")
    print(f"IoU:       {mean_metrics[4]:.4f}")
    print(f"Dice:      {mean_metrics[5]:.4f}")

    print(f"\n[INFO] Saved CSV → {out_csv}")


# ---------------------------------------------------
# CLI
# ---------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_csv", type=str, default="metrics/segmentation_metrics.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = SwinUNet(in_channels=3, num_classes=1).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # Load TEST dataset
    test_ds = SegmentationDataset(args.data_root, split="test", image_size=224)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    os.makedirs("metrics", exist_ok=True)

    # Evaluate
    evaluate(model, test_loader, device, args.output_csv)
