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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/swinunet")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.cfg)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------
    # MODEL
    # ---------------------------------------------------
    model = SwinUNet(in_channels=3, num_classes=1).to(device)

    pre_ckpt = cfg["model"].get("pretrained_ckpt", None)
    if pre_ckpt:
        ckpt_path = os.path.join(ROOT, pre_ckpt)
        if os.path.exists(ckpt_path):
            print(f"[INFO] Loading pretrained weights from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=False)

    # ---------------------------------------------------
    # DATASET
    # ---------------------------------------------------
    dcfg = cfg["dataset"]
    root_dir = dcfg["root_dir"]
    img_size = int(dcfg["image_size"])

    print(f"[INFO] Loading dataset from {root_dir}")
    train_ds = SegmentationDataset(root_dir, split="train", image_size=img_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
    )

    # ---------------------------------------------------
    # OPTIMIZER
    # ---------------------------------------------------
    lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    epochs = int(cfg["training"]["epochs"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("\n[INFO] Starting training...\n")
    for epoch in range(epochs):
        loss, dice = train_one_epoch(model, train_loader, optimizer, epoch, args.output_dir)
        print(f"[EPOCH {epoch}] Loss={loss:.4f} | Dice={dice:.4f}")

    print("\n[INFO] Training Complete!")
