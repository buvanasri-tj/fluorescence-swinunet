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
    p.add_argument("--output_dir", type=str, default="checkpoints/seg")
    return p.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.cfg)

    img_size = cfg["dataset"]["image_size"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]

    model_cfg = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = SwinUNet(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"]
    ).to(device)

    # Load pretrained checkpoint
    ckpt = model_cfg.get("pretrained_ckpt", None)
    if ckpt:
        ckpt = os.path.join(ROOT, ckpt)
        if os.path.exists(ckpt):
            print(f"[INFO] Loading pretrained weights: {ckpt}")
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state, strict=False)

    # Dataset
    train_ds = SegmentationDataset(cfg["dataset"]["root_dir"], split="train", image_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"]["learning_rate"])

    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Training starts...")

    for ep in range(epochs):
        loss, dice = train_one_epoch(model, train_loader, optimizer, ep, args.output_dir)
        print(f"Epoch {ep}: loss={loss:.4f}, dice={dice:.4f}")

    print("[INFO] Training done.")
