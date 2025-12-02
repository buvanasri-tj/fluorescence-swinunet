# scripts/predict_segmentation.py
"""
Predict segmentation masks and create high-quality overlays for publication.
- Handles models saved either as state_dict or checkpoint (with "model" key).
- Safely supports grayscale source images (converts to in_channels).
- Resizes predicted mask back to original image size and creates a transparent overlay
  + draws contours for clarity.
- Processes all three color folders (green, red, yellow) if present, or a single
  input directory if you prefer.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as T

# Ensure repo root in path so `models` and `datasets` imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.swinunet import SwinUNet  # assumes this file exists in models/


def get_transform(in_channels, image_size=224):
    """
    Produce the transform used for prediction that matches training preprocessing.
    If input images are single-channel grayscale, we convert to `in_channels` using
    torchvision.transforms.Grayscale(num_output_channels=in_channels).
    """
    normalize_means = [0.5] * in_channels
    normalize_stds = [0.5] * in_channels

    return T.Compose([
        T.Resize((image_size, image_size)),
        T.Grayscale(num_output_channels=in_channels),
        T.ToTensor(),
        T.Normalize(mean=normalize_means, std=normalize_stds),
    ])


def load_image_tensor(path, transform, device):
    """Load an image (any size), convert to grayscale, transform and return tensor on device."""
    img = Image.open(path).convert("L")  # original fluorescence is single-channel
    t = transform(img).unsqueeze(0).to(device)  # shape: (1, C, H, W)
    return t


def save_mask_png(mask_arr, out_path):
    """Save a binary mask (0/255) numpy array as PNG."""
    pil = Image.fromarray(mask_arr.astype(np.uint8))
    pil.save(out_path)


def overlay_mask(image_path, mask_tensor, out_path, alpha=0.35, contour_color=(255, 255, 0)):
    """
    Create an overlay on the original image.
    - image_path: path to original image (keeps original resolution)
    - mask_tensor: predicted probability (1,1,h,w) or (1,h,w) in normalized scale [0..1] or logits
    - out_path: where to save overlay PNG
    """
    # Load original image in RGB at original size
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)  # H x W x 3

    # Convert mask tensor to numpy (H_pred x W_pred)
    if isinstance(mask_tensor, torch.Tensor):
        mask_np = mask_tensor.squeeze().cpu().numpy()
    else:
        mask_np = np.array(mask_tensor)

    # If mask is logits (maybe outside 0..1), apply sigmoid/clipping
    if mask_np.max() > 1.0 or mask_np.min() < 0.0:
        mask_np = 1.0 / (1.0 + np.exp(-mask_np))  # sigmoid

    # Upscale predicted mask to original image size
    h0, w0 = img_np.shape[:2]
    mask_up = cv2.resize(mask_np, (w0, h0), interpolation=cv2.INTER_LINEAR)

    # Binary mask
    bin_mask = (mask_up > 0.5).astype(np.uint8) * 255  # uint8 0/255

    # Transparent color overlay (red)
    red_layer = np.zeros_like(img_np)
    red_layer[..., 0] = 255

    # Blend where mask present
    overlay = img_np.copy().astype(np.float32)
    red_layer = red_layer.astype(np.float32)
    mask_bool = bin_mask.astype(bool)
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + red_layer[mask_bool] * alpha).astype(np.uint8)

    overlay = overlay.astype(np.uint8)

    # Draw contours for clarity
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        cv2.drawContours(overlay, contours, -1, contour_color, thickness=2)

    # Save overlay
    Image.fromarray(overlay).save(out_path)


def predict_folder(model, device, in_dir, out_root, transform, save_masks=True, save_overlays=True):
    """
    Predict for all PNG files in `in_dir`.
    Saves masks to out_root/masks and overlays to out_root/overlays.
    """
    os.makedirs(out_root, exist_ok=True)
    masks_dir = os.path.join(out_root, "masks")
    overlays_dir = os.path.join(out_root, "overlays")
    if save_masks:
        os.makedirs(masks_dir, exist_ok=True)
    if save_overlays:
        os.makedirs(overlays_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(in_dir) if f.lower().endswith((".png", ".jpg", ".tif", ".tiff"))])
    print(f"[INFO] Predicting {len(files)} images from {in_dir}")

    for fname in files:
        inp = os.path.join(in_dir, fname)
        img_tensor = load_image_tensor(inp, transform, device)  # (1, C, 224, 224)

        with torch.no_grad():
            pred = model(img_tensor)      # model should return logits or single-channel output
            # If model returns (B, C, H, W) and C>1, try to reduce to single channel (e.g. class 1)
            if pred.dim() == 4 and pred.shape[1] > 1:
                # assume binary segmentation with two channels (bg, fg)
                pred = pred[:, 1:2, :, :]  # take foreground logits/prob
            # Ensure pred is (1, 1, H, W) or (1, H, W)
            # Convert logits to probabilities
            pred_prob = torch.sigmoid(pred)

        # Save mask (upsampled to original size)
        # Create binary mask uint8
        pred_np = pred_prob.squeeze().cpu().numpy()  # H x W
        # Upsample to original size and threshold
        # We'll use overlay_mask to upsample again; but save a full-size mask as well
        h_orig, w_orig = Image.open(inp).size[::-1]  # PIL size is (W,H) -> flip
        mask_up = cv2.resize(pred_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        bin_mask = (mask_up > 0.5).astype(np.uint8) * 255

        if save_masks:
            save_mask_png(bin_mask, os.path.join(masks_dir, fname))

        if save_overlays:
            overlay_path = os.path.join(overlays_dir, fname)
            # pass the small pred_prob tensor to overlay function (it will upsample internally)
            overlay_mask(inp, pred_prob.squeeze(), overlay_path)

        print(f"[OK] {fname} → saved mask + overlay")


def safe_load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint flexibly: support full checkpoint dict or bare state_dict"""
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state
    # try strict load first, fallback to non-strict
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[WARN] strict load failed ({e}), attempting non-strict load")
        model.load_state_dict(state_dict, strict=False)


def find_color_test_folders(data_root, colors=("green", "red", "yellow")):
    """
    Look for <data_root>/<color>/test/images and return dict color->path (only existing ones).
    Also support a direct input folder if data_root is a folder containing images (no colors).
    """
    out = {}
    data_root = Path(data_root)
    # first check for color folders
    found = False
    for c in colors:
        p = data_root / c / "test" / "images"
        if p.is_dir():
            out[c] = str(p)
            found = True

    if not found:
        # fallback: if data_root contains images directly, use it as 'all'
        imgs = list(data_root.glob("*.png")) + list(data_root.glob("*.jpg")) + list(data_root.glob("*.tif"))
        if len(imgs) > 0:
            out["all"] = str(data_root)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Predict segmentation and save overlays")
    p.add_argument("--checkpoint", required=True, help="path to checkpoint (best_model.pth or similar)")
    p.add_argument("--data_root", required=True, help="root folder with color subfolders or images")
    p.add_argument("--output_root", default="predictions", help="where to save masks/overlays")
    p.add_argument("--in_channels", type=int, default=3, help="model input channels (default: 3)")
    p.add_argument("--image_size", type=int, default=224, help="Swin input image size (must match model training)")
    p.add_argument("--device", type=str, default=None, help="cpu or cuda (auto-detect if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    print("[INFO] Creating model...")
    model = SwinUNet(in_channels=args.in_channels, num_classes=1).to(device)

    # Load checkpoint (flexible)
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    safe_load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # Prepare transforms
    transform = get_transform(args.in_channels, image_size=args.image_size)

    # Find folders to process
    folders = find_color_test_folders(args.data_root)
    if not folders:
        print(f"[ERROR] No test folders or images found under {args.data_root}")
        raise SystemExit(1)

    # Process each found folder
    for color, folder in folders.items():
        out_dir = os.path.join(args.output_root, color)
        os.makedirs(out_dir, exist_ok=True)
        predict_folder(model, device, folder, out_dir, transform)

    print("[INFO] Prediction complete.")
