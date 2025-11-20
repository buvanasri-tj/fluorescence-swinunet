import argparse
import torch
from torch.utils.data import DataLoader
from datasets.dataset_fluo import FluoDataset
from models.swinunet import SwinUNet
from scripts.trainer import train_one_epoch
import yaml
import os

def parse_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = parse_config(args.cfg)

    train_list = cfg["train_list"]
    val_list = cfg["val_list"]
    batch = cfg["batch_size"]
    img_size = cfg["img_size"]
    lr = cfg["lr"]
    epochs = cfg["epochs"]
    outdir = cfg["output"]

    os.makedirs(outdir, exist_ok=True)

    print("Loading model...")
    model = SwinUNet(num_classes=1).cuda()

    print("Preparing datasets...")
    train_ds = FluoDataset(train_list, img_size)
    val_ds = FluoDataset(val_list, img_size)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_dice = 0

    for epoch in range(epochs):
        loss, dice = train_one_epoch(model, train_loader, opt, epoch)
        print(f"Epoch {epoch}/{epochs} | Training Dice={dice:.4f}")

        torch.save(model.state_dict(), f"{outdir}/epoch_{epoch}.pth")

        if dice > best_dice:
            best_dice = dice
            torch.save(model.state_dict(), f"{outdir}/best_model.pth")
            print("Saved best model.")

    print("Training Completed.")
