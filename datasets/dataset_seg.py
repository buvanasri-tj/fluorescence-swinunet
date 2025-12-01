# datasets/dataset_seg.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    """
    Dataset that reads:
      root/train/images/*.png
      root/train/masks/*.png
    OR
      root/val/images/*.png
      root/val/masks/*.png
    """

    def __init__(self, root_dir, split="train", image_size=256):
        self.root_dir = root_dir
        self.split = split
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")

        self.image_size = image_size

        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(".png")
        ])

        # transforms
        self.transform_img = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.transform_mask = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

        print(f"[INFO] Loaded {len(self.images)} samples for split '{split}' from {root_dir}.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        # convert {0,255} → {0,1}
        mask = (mask > 0.5).float()

        return img, mask
