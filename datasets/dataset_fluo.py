import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class FluoDataset(Dataset):
    """
    Standard fluorescence microscopy segmentation dataset.
    Expects list file lines in this format:

        path/to/image.png path/to/mask.png   (for train and val)
        path/to/image.png                    (for test)

    - Works on Windows and Linux paths
    - Ensures mask is binary {0,1}
    - Converts images to grayscale
    """

    def __init__(self, list_path, img_size=224, is_test=False):
        self.img_size = img_size
        self.is_test = is_test

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")

        self.samples = []

        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                if self.is_test:
                    # test set: images only
                    img = parts[0].replace("\\", "/")
                    self.samples.append((img, None))
                else:
                    # train/val: paired image and mask
                    if len(parts) != 2:
                        print(f"[WARNING] Skipping malformed line: {line}")
                        continue

                    img, mask = parts
                    img = img.replace("\\", "/")
                    mask = mask.replace("\\", "/")
                    self.samples.append((img, mask))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found in list: {list_path}")

        # image transformations
        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

        # mask transformations (only for train/val)
        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image as grayscale
        img = Image.open(img_path).convert("L")
        img = self.transform_img(img)

        # Test set returns only image
        if self.is_test:
            return img, img_path

        # Load mask
        mask = Image.open(mask_path).convert("L")
        mask = self.transform_mask(mask)

        # Convert mask to binary {0,1}
        mask = (mask > 0.5).float()

        return img, mask
