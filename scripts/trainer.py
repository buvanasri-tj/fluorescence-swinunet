import torch
import torch.nn as nn
import os
import csv
import matplotlib.pyplot as plt
from utils.metrics import dice_score

bce = nn.BCEWithLogitsLoss()


def train_one_epoch(model, loader, optimizer, epoch, out_dir):
    model.train()

    epoch_loss = 0.0
    epoch_dice = 0.0

    for step, (img, mask) in enumerate(loader):
        img = img.cuda()
        mask = mask.cuda()

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


def validate(model, loader):
    model.eval()
    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        for img, mask in loader:
            img = img.cuda()
            mask = mask.cuda()

            pred = model(img)
            bce_loss = bce(pred, mask)
            dice = dice_score(pred, mask)

            loss = 0.6 * bce_loss + 0.4 * (1 - dice)

            total_loss += loss.item()
            total_dice += dice

    return total_loss / len(loader), total_dice / len(loader)


def save_plot(log_path, out_png):
    epochs = []
    train_loss = []
    train_dice = []
    val_loss = []
    val_dice = []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_dice.append(float(row["train_dice"]))
            val_loss.append(float(row["val_loss"]))
            val_dice.append(float(row["val_dice"]))

    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.legend()
    plt.savefig(out_png.replace(".png", "_loss.png"))
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_dice, label="Train Dice")
    plt.plot(epochs, val_dice, label="Val Dice")
    plt.legend()
    plt.savefig(out_png.replace(".png", "_dice.png"))
    plt.close()
