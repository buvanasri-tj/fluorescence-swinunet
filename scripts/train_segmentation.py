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
from scripts.trainer import train_one_epoch, validate, save_plot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/seg")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_yaml(args.cfg)

    # Dataset
    root_dir = cfg["dataset"]["root_dir"]
    img_size = cfg["dataset"]["image_size"]

    # Model params
    in_ch = cfg["model"]["in_channels"]
    n_classes = cfg["model"]["num_classes"]
    pre_ckpt = cfg["model"].get("pretrained_ckpt", None)

    # Training params
    lr = float(cfg["training"]["learning_rate"])
    weight_decay = float(cfg["training"]["weight_decay"])
    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[INFO] Creating model…")
    model = SwinUNet(in_channels=in_ch, num_classes=n_classes).to(device)

    # Load pretrained Swin checkpoint
    if pre_ckpt:
        ckpt_path = os.path.join(ROOT, pre_ckpt)
        if os.path.exists(ckpt_path):
            print(f"[INFO] Loading pretrained weights from {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=False)

    # Datasets
    print(f"[INFO] Loading dataset from {root_dir}")
    train_ds = SegmentationDataset(root_dir, split="train", image_size=img_size)
    val_ds = SegmentationDataset(root_dir, split="val", image_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"[INFO] Loaded {len(train_ds)} training samples")
    print(f"[INFO] Loaded {len(val_ds)} validation samples")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # CSV log
    log_csv = os.path.join(args.output_dir, "training_log.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])

    best_val_dice = 0.0
    resume = None

    # RESUME SUPPORT
    last_ckpt = os.path.join(args.output_dir, "last_epoch.pth")
    if os.path.exists(last_ckpt):
        print(f"[INFO] Resuming from {last_ckpt}")
        resume = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = resume["epoch"] + 1
    else:
        start_epoch = 0

    print("\n[INFO] Starting training...")
    for epoch in range(start_epoch, epochs):

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, epoch, args.output_dir)
        val_loss, val_dice = validate(model, val_loader)

        # CSV log
        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_dice, val_loss, val_dice])

        # Save last epoch
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, last_ckpt)

        # Save every epoch
        torch.save(model.state_dict(), os.path.join(args.output_dir, f"epoch_{epoch}.pth"))

        # Save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            print(f"[INFO] New best Dice {best_val_dice:.4f} saved!")

    # Generate plots
    save_plot(log_csv, os.path.join(args.output_dir, "plots.png"))

    print("\n[INFO] Training finished.")
