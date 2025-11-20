# datasets/dataset_seg.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    """
    Dataset for fluorescence SEGMENTATION.
    Requires that list files contain ONLY image paths:
        data/green/train/images/102.png
    Mask is auto-located at:
        data/green/train/masks/102.png
    """

    def __init__(self, list_path, img_size=224, repo_root=None):
        self.img_size = img_size

        # Detect repo root (parent folder of datasets/)
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.repo_root = repo_root

        # Read paths
        with open(list_path, "r") as f:
            self.image_paths = [
                ln.strip().replace("\\", "/")
                for ln in f
                if ln.strip()
            ]

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image paths found in {list_path}")

        # Transforms
        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    def _mask_from_image(self, img_path):
        """
        Convert an image path like:
            data/green/train/images/102.png
        to the corresponding mask path:
            data/green/train/masks/102.png
        """
        parts = img_path.split("/")
        parts[-2] = "masks"  # replace images → masks
        mask_path = "/".join(parts)
        return mask_path

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_rel = self.image_paths[idx]
        img_abs = os.path.join(self.repo_root, img_rel)

        mask_rel = self._mask_from_image(img_rel)
        mask_abs = os.path.join(self.repo_root, mask_rel)

        if not os.path.exists(img_abs):
            raise FileNotFoundError(f"Image not found: {img_abs}")

        if not os.path.exists(mask_abs):
            raise FileNotFoundError(
                f"Mask not found: {mask_abs}\n"
                f"Image: {img_abs}\n"
                f"Something is wrong with your dataset layout."
            )

        img = Image.open(img_abs).convert("L")
        mask = Image.open(mask_abs).convert("L")

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        mask = (mask > 0.5).float()

        return img, mask
