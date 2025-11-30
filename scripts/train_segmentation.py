import os, sys
# Fix import path to repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from models.swinunet import SwinUNet
from datasets.dataset_seg import SegmentationDataset
from scripts.trainer import SegmentationTrainer
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

    resume_path = "checkpoints/swinunet_last.pth"

    trainer = SegmentationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        resume_path=resume_path,
        save_path="checkpoints/swinunet_last.pth"
    )

    start_epoch = trainer.start_epoch

    for epoch in range(start_epoch, 50):
        train_losses = [trainer.train_step(b) for b in train_loader]
        val_losses = [trainer.val_step(b) for b in val_loader]

        print(f"Epoch {epoch}: "
              f"Train={sum(train_losses)/len(train_losses):.4f}, "
              f"Val={sum(val_losses)/len(val_losses):.4f}")

        trainer.save_checkpoint(epoch)

if __name__ == "__main__":
    main()
