import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# -------------------------------
# Ensure repo root is in path
# -------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.swinunet import SwinUNet

# ------------------------------------------
# Preprocessing (3-channel, same as training)
# ------------------------------------------
transform_img = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),   # FIX ✔
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def load_image(path):
    img = Image.open(path).convert("L")
    img = transform_img(img)
    return img.unsqueeze(0)  # (1, 3, 224, 224)

def save_mask(pred_tensor, out_path):
    mask = (pred_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(out_path)

def overlay_mask(image_path, mask_path, out_path):
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    img = np.array(img)
    mask = np.array(mask)

    overlay = img.copy()
    overlay[mask > 128] = [255, 0, 0]  # red overlay

    Image.fromarray(overlay).save(out_path)

def predict_folder(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".png")])

    print(f"[INFO] Predicting {len(files)} images from {input_dir}")

    for fname in files:
        inp = os.path.join(input_dir, fname)
        img_tensor = load_image(inp).cuda()

        with torch.no_grad():
            pred = torch.sigmoid(model(img_tensor))  # output (1, 1, H, W)

        mask_path = os.path.join(output_dir, "masks", fname)
        overlay_path = os.path.join(output_dir, "overlays", fname)

        save_mask(pred, mask_path)
        overlay_mask(inp, mask_path, overlay_path)

        print(f"[OK] {fname} saved")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_root", type=str, default="predictions_seg")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("[INFO] Loading model...")
    model = SwinUNet(in_channels=3, num_classes=1).cuda()

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(state, strict=True)
    model.eval()

    # Predict on each color channel
    for color in ["green", "red", "yellow"]:
        input_dir = os.path.join(args.data_root, color, "test", "images")
        output_dir = os.path.join(args.output_root, color)

        if not os.path.isdir(input_dir):
            print(f"[WARN] {input_dir} missing — skipping.")
            continue

        predict_folder(model, input_dir, output_dir)

    print("[INFO] Finished.")
