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

    # -------------------------
    # Model config
    # -------------------------
    mcfg = cfg["model"]
    in_ch = int(mcfg.get("in_channels", 1))
    n_classes = int(mcfg.get("num_classes", 1))
    img_size = int(cfg["dataset"].get("image_size", 224))

    print("\n[INFO] Creating model…")
    model = SwinUNet(in_channels=in_ch,
                     num_classes=n_classes,
                     img_size=img_size).to(device)

    # -------------------------
    # Pretrained weights
    # -------------------------
    if mcfg.get("pretrained_ckpt"):
        pre_ckpt = os.path.join(ROOT, mcfg["pretrained_ckpt"])
        if os.path.exists(pre_ckpt):
            print(f"[INFO] Loading pretrained weights from {pre_ckpt}")
            state = torch.load(pre_ckpt, map_location=device)
            model.load_state_dict(state, strict=False)
        else:
            print(f"[WARN] Pretrained checkpoint not found: {pre_ckpt}")

    # -------------------------
    # Dataset
    # -------------------------
    dcfg = cfg["dataset"]
    root_dir = dcfg["root_dir"]

    print(f"[INFO] Loading dataset from: {root_dir}")
    train_ds = SegmentationDataset(root_dir, split="train", image_size=img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=2
    )

    print(f"[INFO] Loaded {len(train_ds)} training samples")

    # -------------------------
    # Optimizer
    # -------------------------
    tcfg = cfg["training"]
    lr = float(tcfg["learning_rate"])
    weight_decay = float(tcfg["weight_decay"])
    epochs = int(tcfg["epochs"])

    print(f"[INFO] LR={lr}, WeightDecay={weight_decay}, Epochs={epochs}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=lr,
                                  weight_decay=weight_decay)

    # -------------------------
    # Training
    # -------------------------
    print("\n[INFO] Starting training...")
    for epoch in range(epochs):
        loss, dice = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            args.output_dir
        )
        print(f"[EPOCH {epoch}] Loss={loss:.4f} | Dice={dice:.4f}")

    print("\n[INFO] Training completed.")
