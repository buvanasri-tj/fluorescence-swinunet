import torch
from models.swinunet import SwinUNet
from datasets.dataset_seg import SegmentationDataset
from torch.utils.data import DataLoader
from PIL import Image
import os


def main():
    root = "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SwinUNet(in_channels=3, num_classes=1)
    model.load_state_dict(torch.load("checkpoints/seg_epoch_49.pth", map_location=device))
    model.to(device)
    model.eval()

    test_ds = SegmentationDataset(root, split="test", image_size=256)
    test_loader = DataLoader(test_ds, batch_size=1)

    os.makedirs("predictions", exist_ok=True)

    with torch.no_grad():
        for i, (img, _) in enumerate(test_loader):
            img = img.to(device)
            pred = torch.sigmoid(model(img))[0][0].cpu().numpy() * 255
            Image.fromarray(pred.astype("uint8")).save(f"predictions/{i}.png")


if __name__ == "__main__":
    main()
