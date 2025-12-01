# datasets/dataset_seg.py
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    """
    Loads fluorescence images from:
        data/<color>/<split>/images/
        data/<color>/<split>/masks/

    Stacks green + red + yellow → 3 input channels.
    Uses green mask (or any mask) as ground truth.
    """

    def __init__(self, root_dir, split="train", image_size=224):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        self.colors = ["green", "red", "yellow"]
        self.samples = []

        # Use GREEN masks (same across channels)
        mask_template_dir = os.path.join(root_dir, "green", split, "masks")
        image_template_dir = os.path.join(root_dir, "green", split, "images")

        for fname in sorted(os.listdir(image_template_dir)):
            paths = {}
            good = True

            # collect all 3 channels
            for color in self.colors:
                img_dir = os.path.join(root_dir, color, split, "images", fname)
                if not os.path.exists(img_dir):
                    good = False
                    break
                paths[color] = img_dir

            mask_path = os.path.join(mask_template_dir, fname)
            if not os.path.exists(mask_path):
                good = False

            if good:
                self.samples.append((paths, mask_path))

        print(f"[INFO] Loaded {len(self.samples)} samples ({split})")

        # Transforms
        self.tf_img = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.tf_mask = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, mask_path = self.samples[idx]

        # Load 3-channel fluorescence image
        channels = []
        for color in self.colors:
            img = Image.open(paths[color]).convert("L")
            channels.append(np.array(img))

        # stack → (H,W,3)
        img3 = np.stack(channels, axis=-1)
        img3 = Image.fromarray(img3.astype(np.uint8))

        img_tensor = self.tf_img(img3)

        # Load mask
        mask = Image.open(mask_path).convert("L")
        mask_tensor = self.tf_mask(mask)
        mask_tensor = (mask_tensor > 0.5).float()  # binarize

        return img_tensor, mask_tensor
