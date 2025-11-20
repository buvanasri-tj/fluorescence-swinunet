import os
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    def __init__(self, root, split="train", image_size=256, augment=False):
        self.root = root
        self.split = split
        self.image_size = image_size
        self.augment = augment

        # channel folders
        self.green_img  = os.path.join(root, "green",  split, "images")
        self.red_img    = os.path.join(root, "red",    split, "images")
        self.yellow_img = os.path.join(root, "yellow", split, "images")
        self.mask_dir   = os.path.join(root, "green",  split, "masks")

        self.ids = sorted([
            os.path.splitext(os.path.basename(f))[0]
            for f in glob.glob(os.path.join(self.green_img, "*.png"))
        ])

        self.img_tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

        self.mask_tf = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor()
        ])

    def _load_gray(self, path):
        return Image.open(path).convert("L")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        g = self._load_gray(os.path.join(self.green_img,  f"{id_}.png"))
        r = self._load_gray(os.path.join(self.red_img,    f"{id_}.png"))
        y = self._load_gray(os.path.join(self.yellow_img, f"{id_}.png"))
        m = self._load_gray(os.path.join(self.mask_dir,   f"{id_}.png"))

        g = self.img_tf(g)
        r = self.img_tf(r)
        y = self.img_tf(y)
        m = self.mask_tf(m)

        img = torch.cat([g, r, y], dim=0)

        return img, m
