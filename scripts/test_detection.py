import torch
from models.yolo_detector import YOLODetector
from datasets.dataset_detect import DetectionDataset
from torch.utils.data import DataLoader


def main():
    root = "data"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLODetector(in_channels=3, num_classes=1)
    model.load_state_dict(torch.load("checkpoints/det_epoch_39.pth", map_location=device))
    model.to(device)
    model.eval()

    test_ds = DetectionDataset(root, split="test", image_size=512)
    test_loader = DataLoader(test_ds, batch_size=1)

    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            cls_map, box_map = model(imgs)
            print(cls_map.shape, box_map.shape)


if __name__ == "__main__":
    main()
