import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class DetectionDataset(Dataset):
    def __init__(self, root, split="train", image_size=512):
        self.root = root
        self.split = split
        self.image_size = image_size

        self.green_img  = os.path.join(root, "green",  split, "images")
        self.red_img    = os.path.join(root, "red",    split, "images")
        self.yellow_img = os.path.join(root, "yellow", split, "images")
        self.label_dir  = os.path.join(root, "detection_labels", split)

        self.ids = sorted([
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(self.green_img, "*.png"))
        ])

        self.tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

    def _load_gray(self, p):
        return Image.open(p).convert("L")

    def _load_labels(self, id_):
        p = os.path.join(self.label_dir, id_ + ".txt")
        if not os.path.exists(p):
            return torch.zeros((0,5), dtype=torch.float32)

        boxes = []
        with open(p) as f:
            for line in f:
                c, x, y, w, h = line.strip().split()
                boxes.append([float(c), float(x), float(y), float(w), float(h)])
        return torch.tensor(boxes, dtype=torch.float32)

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
        labels = self._load_labels(id_)

        return img, labels
