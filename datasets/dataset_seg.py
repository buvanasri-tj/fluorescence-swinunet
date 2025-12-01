import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    """
    Loads fluorescence images + masks from:

        data/<color>/<split>/images/
        data/<color>/<split>/masks/

    Supports 3 colors: green, red, yellow.
    """

    def __init__(self, root_dir, split="train", image_size=224):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        self.colors = ["green", "red", "yellow"]
        self.samples = []

        # Scan all color folders
        for color in self.colors:
            img_dir = os.path.join(root_dir, color, split, "images")
            mask_dir = os.path.join(root_dir, color, split, "masks")

            if not os.path.isdir(img_dir):
                print(f"[WARN] Missing: {img_dir}")
                continue

            for fname in sorted(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, fname)
                mask_path = os.path.join(mask_dir, fname)

                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))
                else:
                    print(f"[WARN] Missing mask for {img_path}")

        print(f"[INFO] Loaded {len(self.samples)} samples ({split})")

        # Transforms
        self.t_img = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])

        self.t_mask = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
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

        # convert mask to {0,1}
        mask = (mask > 0.5).float()

        return img, mask
