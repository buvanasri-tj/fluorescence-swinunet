import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class FluoDataset(Dataset):
    """
    Dataset loader for fluorescence PNG images.
    List files contain ONLY image paths:
        data/green/train/images/102.png
    The mask path is automatically inferred as:
        data/green/train/masks/102.png
    """

    def __init__(self, list_path, img_size=224):
        self.img_size = img_size

        # Read list file
        with open(list_path, "r") as f:
            self.image_paths = [ln.strip().replace("\\", "/") for ln in f if ln.strip()]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in list: {list_path}")

        # Verify masks exist
        self.samples = []
        for img_path in self.image_paths:
            mask_path = img_path.replace("/images/", "/masks/")
            if not os.path.exists(img_path):
                print(f"[WARNING] Missing image: {img_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"[WARNING] Missing mask for: {img_path}")
                continue
            self.samples.append((img_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError("No valid image-mask pairs found.")

        # Define transforms
        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        mask = (mask > 0.5).float()  # binarize

        return img, mask
