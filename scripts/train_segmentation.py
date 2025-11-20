# scripts/train_segmentation.py
import torch
from torch.utils.data import DataLoader
from models.swinunet import SwinUNet
from datasets.dataset_seg import SegmentationDataset
from trainer import Trainer

def main():
    train_set = SegmentationDataset("datasets/train_list.txt", img_size=224)
    val_set   = SegmentationDataset("datasets/val_list.txt", img_size=224)

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=4, shuffle=False)

    model = SwinUNet(num_classes=1)
    trainer = Trainer(model, "results/segmentation")

    trainer.train(train_loader, val_loader, epochs=50)

if __name__ == "__main__":
    main()
