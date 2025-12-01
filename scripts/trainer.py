import torch
import torch.nn as nn
import os
import csv
import matplotlib.pyplot as plt
from utils.metrics import dice_score

bce = nn.BCEWithLogitsLoss()


def train_one_epoch(model, loader, optimizer, epoch, out_dir, device):
    model.train()

    epoch_loss = 0.0
    epoch_dice = 0.0

    for step, (img, mask) in enumerate(loader):
        img = img.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        pred = model(img)

        bce_loss = bce(pred, mask)
        dice = dice_score(pred, mask)

        loss = 0.6 * bce_loss + 0.4 * (1 - dice)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_dice += dice

        if step % 20 == 0:
            print(f"Epoch {epoch} | Step {step}/{len(loader)} | "
                  f"Loss: {loss.item():.4f} | Dice: {dice:.4f}")

    return epoch_loss / len(loader), epoch_dice / len(loader)


def validate(model, loader, device):
    if len(loader) == 0:
        print("[WARN] No validation dataset found. Skipping validation.")
        return None, None

    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)

            pred = model(img)

            bce_loss = bce(pred, mask)
            dice = dice_score(pred, mask)

            loss = 0.6 * bce_loss + 0.4 * (1 - dice)

            total_loss += loss.item()
            total_dice += dice

    return total_loss / len(loader), total_dice / len(loader)


def save_plots(csv_path, out_dir):
    if not os.path.exists(csv_path):
        print("[WARN] CSV log missing. Cannot create plots.")
        return

    epochs = []
    train_loss = []
    train_dice = []
    val_loss = []
    val_dice = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_dice.append(float(row["train_dice"]))
            val_loss.append(float(row["val_loss"]))
            val_dice.append(float(row["val_dice"]))

    # LOSS plot
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss_plot.png"))
    plt.close()

    # DICE plot
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_dice, label="Train Dice")
    plt.plot(epochs, val_dice, label="Val Dice")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "dice_plot.png"))
    plt.close()
