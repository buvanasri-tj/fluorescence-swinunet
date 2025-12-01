import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# -----------------------------------
# Ensure repo root is in import path
# -----------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.swinunet import SwinUNet


# -----------------------------------
# Preprocessing
# -----------------------------------
transform_img = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),   # <-- 3-channel input for Swin
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])


def load_image(path):
    img = Image.open(path).convert("L")
    img = transform_img(img)
    return img.unsqueeze(0)   # (1, 3, 224, 224)


def save_mask(mask_tensor, out_path):
    mask = (mask_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(out_path)


# -----------------------------------
# FIXED OVERLAY: upscale mask to original image size
# -----------------------------------
def overlay_mask(image_path, mask_path, out_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size  # original resolution (e.g., 1200x1200)

    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((w, h), Image.NEAREST)  # upscale mask back to full res

    img_np = np.array(img)
    mask_np = np.array(mask)

    overlay = img_np.copy()
    overlay[mask_np > 128] = [255, 0, 0]  # red overlay

    Image.fromarray(overlay).save(out_path)


# -----------------------------------
# Prediction loop
# -----------------------------------
def predict_folder(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    print(f"[INFO] Predicting {len(files)} images from {input_dir}")

    for fname in files:
        inp = os.path.join(input_dir, fname)

        img_tensor = load_image(inp).cuda()

        with torch.no_grad():
            pred = model(img_tensor)     # raw logits
            pred = torch.sigmoid(pred)   # convert to probability

        mask_path = os.path.join(output_dir, "masks", fname)
        overlay_path = os.path.join(output_dir, "overlays", fname)

        save_mask(pred, mask_path)
        overlay_mask(inp, mask_path, overlay_path)

        print(f"[OK] {fname} → mask + overlay")


# -----------------------------------
# CLI
# -----------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_root", type=str, default="predictions_seg")
    return p.parse_args()


# -----------------------------------
# Main
# -----------------------------------
if __name__ == "__main__":
    args = parse_args()

    print("[INFO] Loading model...")
    model = SwinUNet(in_channels=3, num_classes=1).cuda()

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cuda")
    model.load_state_dict(state, strict=True)
    model.eval()

    # predict all three colors
    splits = ["green", "yellow", "red"]

    for color in splits:
        input_dir = os.path.join(args.data_root, color, "test", "images")
        output_dir = os.path.join(args.output_root, color)

        if not os.path.exists(input_dir):
            print(f"[WARN] Missing: {input_dir}, skipping")
            continue

        predict_folder(model, input_dir, output_dir)

    print("\n[INFO] Prediction complete.")
