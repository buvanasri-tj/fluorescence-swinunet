import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from datasets.dataset_fluo import FluoDataset
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from utils.data_utils import overlay_mask_on_image


TEST_LIST = "datasets/test_list.txt"
IMG_SIZE = 224
BATCH_SIZE = 1
CHECKPOINT = "results/checkpoints/epoch_50.pth"   # ← change if needed

SAVE_DIR = "results/segmentation"


def save_prediction(image, mask_pred, name):
    os.makedirs(SAVE_DIR, exist_ok=True)

    mask_pred_img = Image.fromarray(mask_pred.astype(np.uint8) * 255)
    mask_pred_img.save(os.path.join(SAVE_DIR, f"{name}_mask.png"))

    overlay = overlay_mask_on_image(image, mask_pred)
    overlay.save(os.path.join(SAVE_DIR, f"{name}_overlay.png"))


def inference(model, loader):
    model.eval()

    with torch.no_grad():
        for idx, (img, _) in enumerate(loader):
            img_cuda = img.cuda()
            pred = model(img_cuda)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

            mask_pred = (pred > 0.5).astype(np.uint8)

            img_np = img[0].squeeze().numpy()

            save_prediction(img_np, mask_pred, f"sample_{idx}")


if __name__ == "__main__":
    print("Loading test dataset...")
    test_dataset = FluoDataset(
        TEST_LIST,
        img_size=IMG_SIZE,
        allow_missing_mask=True  # ← important for real test images (no masks)
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading model...")
    model = SwinTransformerSys(img_size=IMG_SIZE, num_classes=1)
    model.cuda()

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model.load_state_dict(torch.load(CHECKPOINT))
    print(f"Loaded checkpoint: {CHECKPOINT}")

    print("Running inference...")
    inference(model, test_loader)

    print(f"Predictions saved in: {SAVE_DIR}")
