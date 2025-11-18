# datasets/dataset_fluo.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class FluoDataset(Dataset):
    """
    Robust dataset loader for PNG fluorescence microscopy images.
    Expects list files with lines: /path/to/image.png /path/to/mask.png
    Behaviors:
     - Normalizes path separators to POSIX style
     - Makes paths absolute relative to repo root if they are relative
     - Skips samples whose image/mask files are missing (and logs warnings)
     - Optionally attempts a basename search under data/ as a last resort
    """

    def __init__(self, list_path, img_size=224, num_classes=2, repo_root=None, try_basename_search=True):
        self.img_size = img_size
        self.num_classes = num_classes
        self.try_basename_search = try_basename_search

        # Resolve repo root (defaults to parent of datasets/ directory)
        if repo_root is None:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.repo_root = repo_root

        # Read list file
        list_path = os.path.abspath(list_path) if not os.path.isabs(list_path) else list_path
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"List file not found: {list_path}")
        raw_lines = []
        with open(list_path, "r") as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) < 2:
                    # skip malformed line
                    continue
                raw_lines.append((parts[0], parts[1]))

        # Normalize and validate paths
        self.samples = []
        for img_p, mask_p in raw_lines:
            img_p_norm = self._resolve_path(img_p)
            mask_p_norm = self._resolve_path(mask_p)

            if os.path.exists(img_p_norm) and os.path.exists(mask_p_norm):
                self.samples.append((img_p_norm, mask_p_norm))
            else:
                # Try fallback: replace backslashes with slashes then resolve
                img_p_try = self._resolve_path(img_p.replace("\\", "/"))
                mask_p_try = self._resolve_path(mask_p.replace("\\", "/"))
                if os.path.exists(img_p_try) and os.path.exists(mask_p_try):
                    self.samples.append((img_p_try, mask_p_try))
                    continue

                # Try basename search under repo 'data' folder (last resort)
                if self.try_basename_search:
                    img_basename = os.path.basename(img_p)
                    mask_basename = os.path.basename(mask_p)
                    found_img = self._find_in_data_by_basename(img_basename)
                    found_mask = self._find_in_data_by_basename(mask_basename)
                    if found_img and found_mask:
                        self.samples.append((found_img, found_mask))
                        continue

                # If we reach here, sample is missing -- warn and skip
                print(f"[WARNING] Missing sample, skipping. img: {img_p_norm} exists? {os.path.exists(img_p_norm)} | mask: {mask_p_norm} exists? {os.path.exists(mask_p_norm)}")

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples found for list {list_path}. Check paths and list file formatting.")

        # Define transforms
        self.transform_img = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            # using single-channel normalization (0.5 mean/std) as placeholder
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    def _resolve_path(self, path):
        """
        Normalize a path string and return an absolute path.
        If path is relative, join with repo root.
        """
        # normalize separators
        path = path.replace("\\", os.sep).replace("/", os.sep)
        if os.path.isabs(path):
            return os.path.abspath(path)
        # relative -> join with repo root
        candidate = os.path.abspath(os.path.join(self.repo_root, path))
        return candidate

    def _find_in_data_by_basename(self, basename):
        """
        Search under repo_root/data for a file with given basename.
        Returns the first match absolute path or None.
        """
        data_root = os.path.join(self.repo_root, "data")
        if not os.path.isdir(data_root):
            return None
        for root, _, files in os.walk(data_root):
            if basename in files:
                return os.path.join(root, basename)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, mask_path = self.samples[index]

        # Load grayscale PNG fluorescence image
        img = Image.open(img_path).convert("L")

        # Load mask (assumed binary or grayscale)
        mask = Image.open(mask_path).convert("L")

        img = self.transform_img(img)
        mask = self.transform_mask(mask)

        # Convert mask from {0,255} → {0,1}
        mask = (mask > 0.5).float()

        return img, mask
