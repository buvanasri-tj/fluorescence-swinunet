# scripts/train_segmentation.py
import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.swinunet import SwinUNet
from datasets.dataset_seg import SegmentationDataset
from scripts.trainer import train_one_epoch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="results/checkpoints")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # Model
    print("\n[INFO] Creating model...")
    m_cfg = cfg["model"]
    model = SwinUNet(
        in_channels=m_cfg["in_channels"],
        num_classes=m_cfg["num_classes"]
    ).to(device)

    # Dataset
    ds_cfg = cfg["dataset"]
    img_size = ds_cfg["image_size"]
    root_dir = ds_cfg["root_dir"]

    print(f"[INFO] Loading dataset from {root_dir} with size {img_size} ...")

    train_ds = SegmentationDataset(root_dir, split="train", image_size=img_size)
    val_ds = SegmentationDataset(root_dir, split="val", image_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )

    print("\n[INFO] Starting training...")
    for epoch in range(cfg["training"]["epochs"]):
        loss, dice = train_one_epoch(
            model, train_loader, opt, epoch, args.output_dir
        )
        print(f"Epoch {epoch} | Loss={loss:.4f} | Dice={dice:.4f}")

    print("\n[INFO] Training complete!")
