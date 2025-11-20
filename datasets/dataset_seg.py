# datasets/dataset_seg.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegmentationDataset(Dataset):
    """
    Paired segmentation dataset:
    Each line in list file:
        path/to/image.png path/to/mask.png
    """
    def __init__(self, list_path, img_size=224):
        self.samples = []

        with open(list_path, "r") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) != 2:
                    continue
                img, mask = parts
                img = img.replace("\\", "/")
                mask = mask.replace("\\", "/")
                if os.path.exists(img) and os.path.exists(mask):
                    self.samples.append((img, mask))
                else:
                    print(f"[WARN] Missing pair → {img} | {mask}")

        if len(self.samples) == 0:
            raise RuntimeError("No valid segmentation samples found.")

        self.t_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

        self.t_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = self.t_img(img)
        mask = self.t_mask(mask)
        mask = (mask > 0.5).float()

        return img, mask
