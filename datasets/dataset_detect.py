# Simple YOLO-style dataset loader for bounding-box detection
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class DetectionDataset(Dataset):
    def __init__(self, list_file, transform=None):
        self.items = [x.strip() for x in open(list_file)]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path = self.items[idx]
        label_path = img_path.replace("images", "labels").replace(".png", ".txt")

        img = Image.open(img_path).convert("RGB")

        boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.split())
                    boxes.append([cls, x, y, w, h])

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(boxes), img_path
