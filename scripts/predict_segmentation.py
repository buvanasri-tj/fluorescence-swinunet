import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# -------------------------------
# FIXED: ensure repo root is in path
# -------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.swinunet import SwinUNet


# ------------------------------------------
# Preprocessing
# ------------------------------------------
transform_img = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5], std=[0.5])
])


def load_image(path):
    img = Image.open(path).convert("L")
    return transform_img(img).unsqueeze(0)   # (1, 1, 224, 224)


def save_mask(mask_tensor, out_path):
    """
    mask_tensor: predicted probability map (1, 1, H, W)
    """
    mask = (mask_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(out_path)


def overlay_mask(image_path, mask_path, out_path):
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    mask = np.array(mask)
    img = np.array(img)

    # red overlay
    overlay = img.copy()
    overlay[mask > 128] = [255, 0, 0]

    Image.fromarray(overlay).save(out_path)


# ------------------------------------------
# Prediction
# ------------------------------------------
def predict_folder(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])

    print(f"[INFO] Found {len(files)} images")

    for fname in files:
        inp = os.path.join(input_dir, fname)

        img_tensor = load_image(inp).cuda()

        with torch.no_grad():
            pred = model(img_tensor)        # (1, 1, 224, 224)
            pred = torch.sigmoid(pred)

        mask_path = os.path.join(output_dir, "masks", fname)
        overlay_path = os.path.join(output_dir, "overlays", fname)

        save_mask(pred, mask_path)
        overlay_mask(inp, mask_path, overlay_path)

        print(f"[OK] {fname} → mask + overlay saved")


# ------------------------------------------
# CLI
# ------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="predictions")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("[INFO] Loading model...")
    model = SwinUNet(in_channels=3, num_classes=1).cuda()


    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(state, strict=True)
    model.eval()

    print("[INFO] Starting prediction...")
    predict_folder(model, args.input_dir, args.output_dir)

    print("[INFO] Finished.")
