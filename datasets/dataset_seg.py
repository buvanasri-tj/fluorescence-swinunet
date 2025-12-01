import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    """
    Robust SegmentationDataset that builds 3-channel inputs by stacking:
      green, red, yellow

    - Scans union of filenames across color image folders.
    - If a color image is missing for a filename, fills that channel with zeros.
    - Uses the first available mask in order: green, red, yellow (configurable).
    - Skips samples that have no mask (cannot train without GT).

    Args:
        root_dir: path to data/ (contains green/, red/, yellow/)
        split: "train" / "val" / "test"
        image_size: int (e.g. 224)
    """

    def __init__(self, root_dir, split="train", image_size=224):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        self.colors = ["green", "red", "yellow"]
        self.samples = []  # list of tuples: (dict(color->img_path_or_None), mask_path)

        # Collect filenames (union) for each color
        color_files = {}
        for color in self.colors:
            images_dir = os.path.join(root_dir, color, split, "images")
            if os.path.isdir(images_dir):
                color_files[color] = set(os.listdir(images_dir))
            else:
                color_files[color] = set()
                print(f"[WARN] Missing images dir: {images_dir}")

        # Union of all filenames
        all_fnames = set().union(*color_files.values())

        # Build samples: for each fname, find image paths (or None) and a mask path (if exists)
        missing_mask_count = 0
        for fname in sorted(all_fnames):
            img_paths = {}
            for color in self.colors:
                p = os.path.join(root_dir, color, split, "images", fname)
                img_paths[color] = p if os.path.exists(p) else None

            # prefer mask in green -> red -> yellow
            mask_path = None
            for color in self.colors:
                mp = os.path.join(root_dir, color, split, "masks", fname)
                if os.path.exists(mp):
                    mask_path = mp
                    break

            if mask_path is None:
                missing_mask_count += 1
                # skip sample if no mask available
                continue

            self.samples.append((img_paths, mask_path))

        print(f"[INFO] Found filenames union: {len(all_fnames)}")
        for color in self.colors:
            print(f"[INFO] {color} images: {len(color_files[color])}")
        print(f"[INFO] Skipped (no mask): {missing_mask_count}")
        print(f"[INFO] Loaded {len(self.samples)} samples ({split})")

        # Transforms
        # Note: for the stacked 3-channel image we use RGB-style normalization.
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

    def _load_channel_array(self, path):
        """Return numpy array HxW for a grayscale image at path, or None if path is None."""
        if path is None:
            return None
        try:
            img = Image.open(path).convert("L")
            return np.array(img)
        except Exception as e:
            print(f"[WARN] Could not load {path}: {e}")
            return None

    def __getitem__(self, idx):
        img_paths, mask_path = self.samples[idx]

        # load each channel; if missing, use zeros
        channel_arrays = []
        base_shape = None
        for color in self.colors:
            arr = self._load_channel_array(img_paths[color])
            if arr is None:
                channel_arrays.append(None)
            else:
                channel_arrays.append(arr)
                if base_shape is None:
                    base_shape = arr.shape

        # if no channel had data (shouldn't happen), create zeros of square size
        if base_shape is None:
            h = w = self.image_size
            base_shape = (h, w)

        # build stacked image H x W x 3, filling missing with zeros
        stacked = []
        for arr in channel_arrays:
            if arr is None:
                stacked.append(np.zeros(base_shape, dtype=np.uint8))
            else:
                # if arr has different size, resize via PIL to base_shape
                if arr.shape != base_shape:
                    pil = Image.fromarray(arr)
                    pil = pil.resize((base_shape[1], base_shape[0]), resample=Image.BILINEAR)
                    arr = np.array(pil)
                stacked.append(arr.astype(np.uint8))

        img3 = np.stack(stacked, axis=-1)  # H W 3
        img_pil = Image.fromarray(img3)

        img_tensor = self.tf_img(img_pil)

        # load mask
        mask_pil = Image.open(mask_path).convert("L")
        mask_tensor = self.tf_mask(mask_pil)
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor
