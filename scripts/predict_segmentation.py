# scripts/predict_segmentation.py
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as T

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.swinunet import SwinUNet


# -------------------------------------
# Preprocessing
# -------------------------------------
transform_img = T.Compose([
    T.Resize((224, 224)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])


def load_image(path):
    img = Image.open(path).convert("L")
    img = transform_img(img)
    return img.unsqueeze(0)  # (1,3,224,224)


# ------------------------------------------------------------
# SMOOTH OVERLAY (BILINEAR UPSCALE + GAUSSIAN + ALPHA BLEND)
# ------------------------------------------------------------
def overlay_mask(image_path, mask_path, out_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # load low-res mask
    mask = Image.open(mask_path).convert("L")

    # upscale smoothly (NO blockiness)
    mask = mask.resize((w, h), Image.BILINEAR)

    # soften jagged edges
    mask = mask.filter(ImageFilter.GaussianBlur(radius=1.2))

    img_np = np.array(img)
    mask_np = np.array(mask).astype(np.float32) / 255.0

    # red color
    red = np.zeros_like(img_np)
    red[:, :, 0] = 255

    alpha = 0.55 * mask_np  # smooth transparency

    overlay = (img_np * (1 - alpha[..., None]) +
               red * alpha[..., None]).astype(np.uint8)

    Image.fromarray(overlay).save(out_path)


def save_mask(pred, out_path):
    mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask).save(out_path)


# ------------------------------------------------------------
# Predict one folder
# ------------------------------------------------------------
def predict_folder(model, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)

    files = [f for f in sorted(os.listdir(input_dir))
             if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"[INFO] Predicting {len(files)} images …")

    for fname in files:
        path = os.path.join(input_dir, fname)

        img_tensor = load_image(path).cuda()

        with torch.no_grad():
            pred = torch.sigmoid(model(img_tensor))

        mask_path = os.path.join(output_dir, "masks", fname)
        overlay_path = os.path.join(output_dir, "overlays", fname)

        save_mask(pred, mask_path)
        overlay_mask(path, mask_path, overlay_path)

        print(f"[OK] {fname}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--output_root", type=str, default="predictions_seg")
    return p.parse_args()


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Creating model…")
    model = SwinUNet(in_channels=3, num_classes=1).to(device)

    print("[INFO] Loading checkpoint:", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    for color in ["green", "yellow", "red"]:
        inp = os.path.join(args.data_root, color, "test", "images")
        out = os.path.join(args.output_root, color)

        if os.path.exists(inp):
            predict_folder(model, inp, out)
        else:
            print(f"[WARN] Missing: {inp}")

    print("\n[INFO] Prediction complete.")
