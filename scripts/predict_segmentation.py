import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# -------------------------------
# Ensure repo root is in sys.path
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
    return transform_img(img).unsqueeze(0)


def save_mask(mask_tensor, out_path):
    mask = (mask_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(out_path)


def overlay_mask(image_path, mask_path, out_path):
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    mask = np.array(mask)
    img = np.array(img)

    overlay = img.copy()
    overlay[mask > 128] = [255, 0, 0]   # red overlay

    Image.fromarray(overlay).save(out_path)


# ------------------------------------------
# Prediction for a single folder
# ------------------------------------------
def predict_folder(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])

    print(f"[INFO] Predicting {len(files)} images from {input_dir}")

    for fname in files:
        img_path = os.path.join(input_dir, fname)
        img_tensor = load_image(img_path).cuda()

        with torch.no_grad():
            pred = model(img_tensor)
            pred = torch.sigmoid(pred)

        mask_out = os.path.join(output_dir, "masks", fname)
        overlay_out = os.path.join(output_dir, "overlays", fname)

        save_mask(pred, mask_out)
        overlay_mask(img_path, mask_out, overlay_out)

        print(f"[OK] {fname} saved.")


# ------------------------------------------
# CLI
# ------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--output_root", type=str, default="predictions_seg")
    return p.parse_args()


# ------------------------------------------
# MAIN
# ------------------------------------------
if __name__ == "__main__":
    args = parse_args()

    print("[INFO] Loading model...")
    model = SwinUNet(in_channels=3, num_classes=1).cuda()

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(state, strict=True)
    model.eval()

    COLORS = ["green", "red", "yellow"]

    for color in COLORS:
        input_dir = os.path.join(args.data_root, color, "test", "images")
        output_dir = os.path.join(args.output_root, color)

        if not os.path.isdir(input_dir):
            print(f"[WARN] Missing: {input_dir}, skipping...")
            continue

        predict_folder(model, input_dir, output_dir)

    print("[INFO] Finished all predictions.")
