import torch
from torch.utils.data import DataLoader
from models.swinunet import SwinUNet
from datasets.dataset_seg import SegmentationDataset
from scripts.trainer import SegmentationTrainer
import torch.nn.functional as F
import torch.nn as nn


def main():
    root = "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = SegmentationDataset(root, split="train", image_size=256)
    val_ds   = SegmentationDataset(root, split="val", image_size=256)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=4)

    model = SwinUNet(in_channels=3, num_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    trainer = SegmentationTrainer(model, optimizer, criterion, device)

    for epoch in range(50):
        train_losses = []
        for batch in train_loader:
            loss = trainer.train_step(batch)
            train_losses.append(loss)

        val_losses = []
        for batch in val_loader:
            loss = trainer.val_step(batch)
            val_losses.append(loss)

        print(f"Epoch {epoch}: Train {sum(train_losses)/len(train_losses):.4f}, "
              f"Val {sum(val_losses)/len(val_losses):.4f}")

        torch.save(model.state_dict(), f"checkpoints/seg_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()
