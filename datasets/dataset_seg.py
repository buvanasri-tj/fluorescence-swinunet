import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    """
    Loads fluorescence images from:
        data/<color>/<split>/images/*.png
        data/<color>/<split>/masks/*.png

    Colors = ["green", "red", "yellow"]
    """

    def __init__(self, root_dir, split="train", image_size=224):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        self.colors = ["green", "red", "yellow"]
        self.samples = []

        # ---------------------------
        # LOAD IMAGES FOR EACH COLOR
        # ---------------------------
        for color in self.colors:
            img_dir = os.path.join(root_dir, color, split, "images")
            mask_dir = os.path.join(root_dir, color, split, "masks")

            if not os.path.isdir(img_dir):
                print(f"[WARN] Missing folder: {img_dir}")
                continue

            for fname in sorted(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, fname)
                mask_path = os.path.join(mask_dir, fname)

                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))
                else:
                    print(f"[WARN] No mask for: {img_path}")

        print(f"[INFO] Loaded {len(self.samples)} samples ({split})")

        # --------------------------
        # TRANSFORMS
        # --------------------------
        self.transform_img = T.Compose([
            T.Resize((image_size, image_size)),
            T.Grayscale(num_output_channels=3),   # 🔥 convert 1-channel → 3-channel
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

        self.transform_mask = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("L")     # grayscale input
        mask = Image.open(mask_path).convert("L")   # binary mask

        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        mask = (mask > 0.5).float()

        return img, mask
