import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    """
    Custom dataset class for PNG fluorescence microscopy images.
    Reads (image, mask) pairs from train_list.txt or test_list.txt.
    """

    def __init__(self, list_path, img_size=224, num_classes=2):
        self.img_size = img_size
        self.num_classes = num_classes

        with open(list_path, "r") as f:
            self.samples = [line.strip().split() for line in f.readlines()]

        # Basic augmentations
        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]

        # Load grayscale PNG fluorescence image
        img = Image.open(img_path).convert("L")

        # Load binary mask
        mask = Image.open(mask_path).convert("L")

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        # Convert mask from {0,255} → {0,1}
        mask = (mask > 0.5).float()

        return img, mask
