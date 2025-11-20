import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import json


class FluorescenceSegmentationDataset(Dataset):
    """
    Segmentation dataset for fluorescence microscopy (PNG).
    Expects list.txt with ONE COLUMN:
        data/green/train/images/102.png
    Mask path is inferred automatically:
        replace 'images' with 'masks'
    """

    def __init__(self, list_path, img_size=224, normalize=True):
        self.list_path = os.path.abspath(list_path)

        if not os.path.exists(self.list_path):
            raise FileNotFoundError(f"List file not found: {self.list_path}")

        with open(self.list_path, "r") as f:
            self.img_paths = [ln.strip().replace("\\", "/") for ln in f if ln.strip()]

        self.img_size = img_size
        self.normalize = normalize

        # image transforms
        if normalize:
            self.transform_img = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            self.transform_img = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
            ])

        # mask transform
        self.transform_mask = T.Compose([
            T.Resize((img_size, img_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    def _mask_from_image(self, img_path):
        return img_path.replace("/images/", "/masks/")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]

        mask_path = self._mask_from_image(img_path)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image missing: {img_path}")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask missing: {mask_path}")

        img = Image.open(img_path).convert("L")      # grayscale
        mask = Image.open(mask_path).convert("L")    # binary mask

        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        mask = (mask > 0.5).float()

        return img, mask



class FluorescenceDetectionDataset(Dataset):
    """
    Detection dataset for YOLO.
    Expects:
        - list.txt → only image paths
        - bounding box annotations in JSON beside each image:
              /images/102.png → /labels/102.json

    JSON format:
       { "boxes": [ [x1, y1, x2, y2], ... ] }
    """

    def __init__(self, list_path, img_size=640):
        self.list_path = os.path.abspath(list_path)

        with open(self.list_path, "r") as f:
            self.img_paths = [ln.strip().replace("\\", "/") for ln in f if ln.strip()]

        self.img_size = img_size
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def _label_from_image(self, img_path):
        basename = os.path.basename(img_path).replace(".png", ".json")
        return img_path.replace("/images/", "/labels/").replace(".png", ".json")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self._label_from_image(img_path)

        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # load bboxes or empty
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                data = json.load(f)
            boxes = data.get("boxes", [])
        else:
            boxes = []

        return img, boxes



class FluorescenceCountingDataset(Dataset):
    """
    Counting dataset.
    Expects:
        images only (one column list)
        count annotations in parallel folder:
            /images/102.png → /counts/102.txt
        count file contains a single number.
    """

    def __init__(self, list_path, img_size=224):
        self.list_path = os.path.abspath(list_path)

        with open(self.list_path, "r") as f:
            self.img_paths = [ln.strip().replace("\\", "/") for ln in f if ln.strip()]

        self.img_size = img_size

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def _count_from_image(self, img_path):
        return img_path.replace("/images/", "/counts/").replace(".png", ".txt")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        count_path = self._count_from_image(img_path)

        img = Image.open(img_path).convert("L")
        img = self.transform(img)

        if not os.path.exists(count_path):
            count = 0
        else:
            with open(count_path, "r") as f:
                count = int(f.read().strip())

        return img, count
