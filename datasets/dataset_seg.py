import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


class SegmentationDataset(Dataset):
    """
    Automatically loads ALL images+masks from:
    data/train/images/
    data/train/masks/
    data/val/images/
    data/val/masks/
    """

    def __init__(self, root_dir, split="train", image_size=256):
        self.root = os.path.join(root_dir, split)
        self.img_dir = os.path.join(self.root, "images")
        self.mask_dir = os.path.join(self.root, "masks")

        self.files = sorted(os.listdir(self.img_dir))

        self.transform_img = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

        self.transform_mask = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        img = self.transform_img(img)
        mask = self.transform_mask(mask)
        mask = (mask > 0.5).float()

        return img, mask
