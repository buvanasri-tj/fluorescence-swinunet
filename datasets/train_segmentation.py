import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.dataset_fluo import FluoDataset
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from utils.metrics import dice_coef

# ----------------------------
# Config
# ----------------------------
TRAIN_LIST = "datasets/train_list.txt"
VAL_LIST   = "datasets/val_list.txt"
IMG_SIZE   = 224
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
CHECKPOINT_DIR = "results/checkpoints"


def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    total_dice = 0

    for imgs, masks in loader:
        imgs = imgs.cuda()
        masks = masks.cuda()

        preds = model(imgs)
        preds = preds.squeeze(1)

        loss = loss_fn(preds, masks.squeeze(1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dice = dice_coef(preds, masks.squeeze(1)).item()

        total_loss += loss.item()
        total_dice += dice

    return total_loss / len(loader), total_dice / len(loader)


def val_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = 0
    total_dice = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.cuda()
            masks = masks.cuda()

            preds = model(imgs)
            preds = preds.squeeze(1)

            loss = loss_fn(preds, masks.squeeze(1))
            dice = dice_coef(preds, masks.squeeze(1)).item()

            total_loss += loss.item()
            total_dice += dice

    return total_loss / len(loader), total_dice / len(loader)


if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("Loading datasets...")
    train_dataset = FluoDataset(TRAIN_LIST, img_size=IMG_SIZE)
    val_dataset   = FluoDataset(VAL_LIST, img_size=IMG_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading model...")
    model = SwinTransformerSys(img_size=IMG_SIZE, num_classes=1)
    model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Training started...")

    for epoch in range(EPOCHS):
        tr_loss, tr_dice = train_epoch(model, train_loader, optimizer, loss_fn)
        vl_loss, vl_dice = val_epoch(model, val_loader, loss_fn)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss={tr_loss:.4f} | Train Dice={tr_dice:.4f} | "
              f"Val Loss={vl_loss:.4f} | Val Dice={vl_dice:.4f}")

        torch.save(model.state_dict(),
                   f"{CHECKPOINT_DIR}/epoch_{epoch+1}.pth")

    print("Training completed!")
