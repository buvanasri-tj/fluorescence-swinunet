import os
import glob
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class CountingDataset(Dataset):
    def __init__(self, root, split="train", count_csv="data/counts.csv", image_size=256):
        self.root = root
        self.split = split
        self.image_size = image_size

        self.green_img  = os.path.join(root, "green",  split, "images")
        self.red_img    = os.path.join(root, "red",    split, "images")
        self.yellow_img = os.path.join(root, "yellow", split, "images")

        self.ids = sorted([
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(self.green_img, "*.png"))
        ])

        self.counts = {}
        with open(count_csv) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.counts[row[0]] = float(row[1])

        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

    def _load_gray(self, p):
        return Image.open(p).convert("L")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        g = self._load_gray(os.path.join(self.green_img,  f"{id_}.png"))
        r = self._load_gray(os.path.join(self.red_img,    f"{id_}.png"))
        y = self._load_gray(os.path.join(self.yellow_img, f"{id_}.png"))

        g = self.tf(g)
        r = self.tf(r)
        y = self.tf(y)

        img = torch.cat([g, r, y], dim=0)
        count = torch.tensor([self.counts[id_]], dtype=torch.float32)

        return img, count
