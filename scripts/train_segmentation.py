import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

# ==========================================================
# BULLETPROOF FIX: Ensure repo root is first in Python path
# ==========================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ==========================================================

from models.swinunet import SwinUNet
from datasets.dataset_seg import SegmentationDataset
from scripts.trainer import train_one_epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--train_list", type=str, default="datasets/train_list.txt")
    parser.add_argument("--val_list", type=str, default="datasets/val_list.txt")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints")
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()

    # Load YAML config
    cfg = load_yaml(args.cfg)

    # Allow overriding some YAML options with CLI flags (if provided)
    if args.img_size is not None:
        cfg["dataset"]["image_size"] = args.img_size
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_epochs is not None:
        cfg["training"]["epochs"] = args.num_epochs

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Instantiate model
    print("\n[INFO] Creating model...")
    m_cfg = cfg.get("model", {})
    in_ch = m_cfg.get("in_channels", 3)
    n_classes = m_cfg.get("num_classes", 1)
    model = SwinUNet(in_channels=in_ch, num_classes=n_classes).to(device)

    # Load pretrained ckpt if present
    pre_ckpt = m_cfg.get("pretrained_ckpt", None)
    if pre_ckpt:
        # Normalize relative paths: if not absolute, make repo-root relative
        if not os.path.isabs(pre_ckpt):
            pre_ckpt = os.path.join(ROOT, pre_ckpt)
        if os.path.exists(pre_ckpt):
            print(f"[INFO] Loading pretrained checkpoint from: {pre_ckpt}")
            state = torch.load(pre_ckpt, map_location=device)
            try:
                model.load_state_dict(state, strict=False)
                print("[INFO] Pretrained weights loaded (strict=False).")
            except Exception as e:
                print(f"[WARN] Could not fully load pretrained weights: {e}")
        else:
            print(f"[WARN] Pretrained checkpoint not found at: {pre_ckpt}")

    # Prepare dataset & dataloader
    ds_cfg = cfg.get("dataset", {})
    root_dir = ds_cfg.get("root_dir", "data/")
    img_size = ds_cfg.get("image_size", 256)
    batch_size = cfg.get("training", {}).get("batch_size", 4)

    print(f"\n[INFO] Using dataset root: {root_dir}, image_size: {img_size}")
    train_ds = SegmentationDataset(root_dir, split="train", image_size=img_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    # Optimizer
    tr_cfg = cfg.get("training", {})
    lr = tr_cfg.get("learning_rate", 1e-4)
    weight_decay = tr_cfg.get("weight_decay", 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    epochs = tr_cfg.get("epochs", 50)
    print("\n[INFO] Starting training")
    for epoch in range(epochs):
        loss, dice = train_one_epoch(model, train_loader, optimizer, epoch, args.output_dir)
        print(f"Epoch {epoch} Completed | Loss={loss:.4f} | Dice={dice:.4f}")

    print("\n[INFO] Training finished.")
