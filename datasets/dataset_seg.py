import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch


class SegmentationDataset(Dataset):
    def __init__(self, root, split="train", image_size=256, augment=False):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment

        # Directories for channels
        self.green_img = os.path.join(root, "green", split, "images")
        self.red_img = os.path.join(root, "red", split, "images")
        self.yellow_img = os.path.join(root, "yellow", split, "images")
        self.green_mask = os.path.join(root, "green", split, "masks")

        # Base on green image folder
        self.ids = sorted([os.path.splitext(os.path.basename(f))[0]
                           for f in glob.glob(os.path.join(self.green_img, "*.png"))])

        self.img_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.ids)

    def _load(self, path):
        return Image.open(path).convert("L")

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        g = self._load(os.path.join(self.green_img, f"{id_}.png"))
        r = self._load(os.path.join(self.red_img, f"{id_}.png"))
        y = self._load(os.path.join(self.yellow_img, f"{id_}.png"))
        m = self._load(os.path.join(self.green_mask, f"{id_}.png"))

        g = self.img_transform(g)
        r = self.img_transform(r)
        y = self.img_transform(y)
        m = self.mask_transform(m)

        rgb = torch.cat([g, r, y], dim=0)

        return rgb, m
