import torch
import torch.nn as nn
import os
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
            print(f"Epoch {epoch} | Step {step}/{len(loader)} "
                  f"| Loss: {loss.item():.4f} | Dice: {dice:.4f}")

    # Save model checkpoint
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, f"epoch_{epoch}.pth"))

    return epoch_loss / len(loader), epoch_dice / len(loader)
