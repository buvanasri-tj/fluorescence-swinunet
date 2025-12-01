import os
import sys
import argparse
import yaml
import torch
from torch.utils.data import DataLoader

# Ensure project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.swinunet import SwinUNet
from datasets.dataset_seg import SegmentationDataset
from scripts.trainer import train_one_epoch, validate, save_plots
import csv


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

    # Make checkpoint folder
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------
    # MODEL
    # ------------------------------
    model = SwinUNet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"]
    ).to(device)

    pre_ckpt = cfg["model"].get("pretrained_ckpt", None)
    if pre_ckpt:
        ckpt_path = os.path.join(ROOT, pre_ckpt)
        if os.path.exists(ckpt_path):
            print(f"[INFO] Loading pretrained Swin weights: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=False)

    # ------------------------------
    # DATASET
    # ------------------------------
    root_dir = cfg["dataset"]["root_dir"]
    img_size = cfg["dataset"]["image_size"]

    print(f"[INFO] Loading dataset from {root_dir}")
    train_ds = SegmentationDataset(root_dir, split="train", image_size=img_size)

    # Load val safely (if missing → empty)
    val_ds = SegmentationDataset(root_dir, split="val", image_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=2)

    val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=2)

    # ------------------------------
    # OPTIMIZER & TRAINING PARAMS
    # ------------------------------
    lr = float(cfg["training"]["learning_rate"])
    wd = float(cfg["training"]["weight_decay"])
    epochs = int(cfg["training"]["epochs"])
    batch_size = int(cfg["training"]["batch_size"])


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # ------------------------------
    # CSV LOGGING
    # ------------------------------
    csv_path = os.path.join(args.output_dir, "log.csv")

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])

    # ------------------------------
    # RESUME TRAINING IF POSSIBLE
    # ------------------------------
    last_ckpt = os.path.join(args.output_dir, "last_epoch.pth")
    best_ckpt = os.path.join(args.output_dir, "best_model.pth")

    start_epoch = 0
    best_dice = 0.0

    if os.path.exists(last_ckpt):
        print(f"[INFO] Resuming from last checkpoint")
        chk = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(chk["model"])
        optimizer.load_state_dict(chk["optimizer"])
        start_epoch = chk["epoch"] + 1
        best_dice = chk.get("best_dice", 0.0)

    # ------------------------------
    # TRAINING LOOP
    # ------------------------------
    print("\n[INFO] Starting training...\n")

    for epoch in range(start_epoch, epochs):

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, epoch, args.output_dir, device)
        val_loss, val_dice = validate(model, val_loader, device)

        # Write CSV row
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, train_dice, val_loss, val_dice])

        # Save last epoch (resume)
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_dice": best_dice
        }, last_ckpt)

        # Save every epoch
        torch.save(model.state_dict(),
                   os.path.join(args.output_dir, f"epoch_{epoch}.pth"))

        # Best checkpoint
        if val_dice is not None and val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_ckpt)
            print(f"[INFO] ✔ New Best Dice {best_dice:.4f}")

    # ------------------------------
    # FINAL PLOTS
    # ------------------------------
    save_plots(csv_path, args.output_dir)

    print("\n[INFO] Training Finished Successfully!")
