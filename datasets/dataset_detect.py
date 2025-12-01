# datasets/detection_dataset.py
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

def collate_fn(batch):
    return tuple(zip(*batch))

class DetectionDataset(Dataset):
    """
    DetectionDataset supports:
    - COCO-style JSON annotation file (key: "images", "annotations")
      Put the json file in annotation_path (exact path).
    - Simple text annotations per image:
        labels_dir/<image_basename>.txt
      Each line: x1 y1 x2 y2 class_id

    Args:
        images_dir: path to images folder
        labels_dir_or_json: path to labels folder OR coco json file
        transforms: torchvision transforms applied to PIL image (returns tensor)
        classes_map: dict mapping class ids to names (optional)
    """
    def __init__(self, images_dir, labels_dir_or_json, transforms=None, classes_map=None, img_exts=None):
        self.images_dir = images_dir
        self.labels = labels_dir_or_json
        self.transforms = transforms
        self.classes_map = classes_map or {}
        self.img_exts = img_exts or [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

        self.use_coco = False
        self.records = []  # list of (img_path, annotations list)

        if os.path.isfile(self.labels) and self.labels.lower().endswith(".json"):
            # parse COCO
            self.use_coco = True
            with open(self.labels, "r") as f:
                coco = json.load(f)
            imgs = {i["id"]: i for i in coco.get("images", [])}
            anns = {}
            for a in coco.get("annotations", []):
                img_id = a["image_id"]
                anns.setdefault(img_id, []).append(a)
            for img_id, img_meta in imgs.items():
                fname = img_meta["file_name"]
                img_path = os.path.join(self.images_dir, fname)
                if not os.path.exists(img_path):
                    continue
                self.records.append((img_path, anns.get(img_id, [])))
        else:
            # labels_dir expected to contain one txt per image OR a single file listing images
            labels_dir = self.labels
            if not os.path.isdir(labels_dir):
                raise FileNotFoundError(f"Labels folder not found: {labels_dir}")
            # gather images
            for fname in sorted(os.listdir(self.images_dir)):
                if not any(fname.lower().endswith(e) for e in self.img_exts):
                    continue
                img_path = os.path.join(self.images_dir, fname)
                base = os.path.splitext(fname)[0]
                label_file = os.path.join(labels_dir, base + ".txt")
                ann = []
                if os.path.exists(label_file):
                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 4:
                                continue
                            x1, y1, x2, y2 = map(float, parts[:4])
                            cls = int(parts[4]) if len(parts) >= 5 else 1
                            ann.append({"bbox": [x1, y1, x2, y2], "category_id": cls})
                # even if no ann, include (useful for test)
                self.records.append((img_path, ann))

        # default transform if None: convert to tensor only
        if self.transforms is None:
            self.transforms = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_path, ann = self.records[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        if self.use_coco:
            for a in ann:
                # COCO bbox is [x,y,w,h]
                x, y, bw, bh = a["bbox"]
                x2 = x + bw
                y2 = y + bh
                boxes.append([x, y, x2, y2])
                labels.append(a.get("category_id", 1))
                areas.append(bw * bh)
                iscrowd.append(a.get("iscrowd", 0))
        else:
            for a in ann:
                x1, y1, x2, y2 = a["bbox"]
                boxes.append([x1, y1, x2, y2])
                labels.append(a.get("category_id", 1))
                areas.append((x2 - x1) * (y2 - y1))
                iscrowd.append(0)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([idx])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target
