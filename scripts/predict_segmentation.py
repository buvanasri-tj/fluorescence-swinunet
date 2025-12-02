# scripts/predict_segmentation.py
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import cv2

# Ensure repo root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.swinunet import SwinUNet

# -------------------------
# Utilities & transforms
# -------------------------
def get_transform(in_chans):
    """Return transform that produces the correct number of channels for the model."""
    # We always resize to 224 for Swin input (model trained with 224)
    # Normalize like training: mean=0.5 std=0.5 for each channel
    if in_chans == 3:
        return T.Compose([
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3),  # convert 1->3 if needed
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])
    else:
        # if model expects 1 channel (rare for Swin) keep single channel
        return T.Compose([
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])


def preprocess_for_model(image_path, transform):
    """Load original image (PIL), return original PIL and transformed tensor (1 x C x H x W)."""
    img = Image.open(image_path).convert("L")  # read as single-channel from microscope
    orig = Image.open(image_path).convert("RGB")  # original RGB for overlay
    tensor = transform(img).unsqueeze(0)  # add batch dim
    return orig, tensor


def postprocess_mask(prob_map, orig_size, morph_kernel=3, blur_ksize=5):
    """
    prob_map: numpy array HxW with probabilities (0..1) at model input resolution (224x224)
    orig_size: (W,H) of original image to resize mask back to full-res
    returns: cleaned binary mask as PIL L (0/255)
    """
    # Convert prob -> binary (threshold)
    mask = (prob_map >= 0.5).astype(np.uint8) * 255  # 0/255

    # morphological closing to fill holes + opening to remove tiny specks
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Optional blurring to smooth edges (apply before resizing gives better result)
    if blur_ksize > 0:
        # blur expects odd kernel
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    # Resize to original size (NEAREST to keep binary), but blurring above creates near-binary which smooths
    orig_w, orig_h = orig_size
    mask_resized = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Ensure strict binary again
    _, mask_resized = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

    return Image.fromarray(mask_resized.astype(np.uint8))


def create_alpha_overlay(orig_pil, mask_pil, color=(255, 0, 0), alpha=0.55):
    """
    orig_pil: RGB PIL
    mask_pil: L PIL (0/255)
    color: RGB tuple for overlay color
    alpha: blending factor for overlayed color
    returns: PIL RGB blended image
    """
    orig_np = np.array(orig_pil).astype(np.uint8)
    mask_np = np.array(mask_pil).astype(np.uint8)  # 0/255

    overlay = orig_np.copy()
    colored = np.zeros_like(orig_np)
    colored[:, :] = color

    mask_bool = mask_np > 0
    # blend only where mask is True
    overlay[mask_bool] = (overlay[mask_bool].astype(float) * (1.0 - alpha) +
                          colored[mask_bool].astype(float) * alpha).astype(np.uint8)

    return Image.fromarray(overlay)


# -----------------------------------
# Prediction loop
# -----------------------------------
def predict_folder(model, input_dir, output_dir, transform, device,
                   morph_kernel=3, blur_ksize=5, alpha=0.55):
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    overlays_dir = os.path.join(output_dir, "overlays")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)

    images = sorted([f for f in os.listdir(input_dir)
                     if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])

    print(f"[INFO] Predicting {len(images)} images from {input_dir}")

    for fname in images:
        img_path = os.path.join(input_dir, fname)

        orig_pil, input_tensor = preprocess_for_model(img_path, transform)
        input_tensor = input_tensor.to(device)

        with torch.no_grad():
            logits = model(input_tensor)           # raw logits (B, C, H, W)
            probs = torch.sigmoid(logits)          # probs 0..1
            # if model outputs multi-channel (C>1) but we used num_classes=1 - handle accordingly
            probs = probs.squeeze().cpu().numpy()  # (H, W) if 1-channel or (C, H, W) otherwise

            # If multi-channel returned, assume first channel is foreground
            if probs.ndim == 3:
                probs = probs[0]

        # probs currently at model input resolution (224,224) — postprocess then resize to original
        mask_pil = postprocess_mask(probs, orig_pil.size, morph_kernel=morph_kernel, blur_ksize=blur_ksize)

        # Save mask (binary) and overlay (alpha-blend)
        mask_out = os.path.join(masks_dir, fname)
        overlay_out = os.path.join(overlays_dir, fname)

        mask_pil.save(mask_out)
        overlay = create_alpha_overlay(orig_pil, mask_pil, alpha=alpha)
        overlay.save(overlay_out)

        print(f"[OK] {fname} — mask & overlay saved.")


# -----------------------------------
# CLI
# -----------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="path to model checkpoint (.pth)")
    p.add_argument("--data_root", type=str, required=True,
                   help="root data directory containing color folders (green, yellow, red) with test/images")
    p.add_argument("--output_root", type=str, default="predictions_seg",
                   help="root output folder to store masks/overlays per color")
    p.add_argument("--in_channels", type=int, default=3,
                   help="model input channels (3 for Swin trained with 3-channel input)")
    p.add_argument("--colors", type=str, default="green,yellow,red",
                   help="comma-separated color folders to predict")
    p.add_argument("--device", type=str, default=None,
                   help="device to run on (cuda/cpu). auto-detect if omitted")
    p.add_argument("--morph_kernel", type=int, default=3,
                   help="morphological kernel size to clean masks")
    p.add_argument("--blur_ksize", type=int, default=5,
                   help="gaussian blur kernel to smooth mask boundaries (odd int, 0 to disable)")
    p.add_argument("--alpha", type=float, default=0.55,
                   help="overlay alpha for blending mask color")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("[INFO] Creating model...")
    model = SwinUNet(in_channels=args.in_channels, num_classes=1).to(device)

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    chk = torch.load(args.checkpoint, map_location=device)
    # load state_dict robustly in case shapes differ slightly
    if "state_dict" in chk:
        state = chk["state_dict"]
    elif "model" in chk:
        state = chk["model"]
    else:
        state = chk
    model.load_state_dict(state, strict=False)
    model.eval()

    transform = get_transform(args.in_channels)

    colors = [c.strip() for c in args.colors.split(",") if c.strip()]

    for color in colors:
        input_dir = os.path.join(args.data_root, color, "test", "images")
        if not os.path.isdir(input_dir):
            print(f"[WARN] Missing test folder for color '{color}': {input_dir} — skipped")
            continue

        out_dir = os.path.join(args.output_root, color)
        predict_folder(model, input_dir, out_dir, transform, device,
                       morph_kernel=args.morph_kernel,
                       blur_ksize=args.blur_ksize,
                       alpha=args.alpha)

    print("\n[INFO] All done.")
