import os
import cv2
import numpy as np
from glob import glob

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_overlay(img_path, mask_path, out_path, alpha=0.5):
    """Create segmentation overlay (mask blended on top of image)."""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None or mask is None:
        print(f"[WARN] Missing image or mask: {img_path}")
        return

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Create a red mask overlay
    mask_color = np.zeros_like(img_color)
    mask_color[:, :, 2] = mask   # Red channel = mask

    blended = cv2.addWeighted(img_color, 1 - alpha, mask_color, alpha, 0)

    cv2.imwrite(out_path, blended)

def process_folder(images_dir, masks_dir, out_dir):
    ensure_dir(out_dir)

    images = sorted(glob(os.path.join(images_dir, "*.png")))

    for img_path in images:
        fname = os.path.basename(img_path)

        mask_path = os.path.join(masks_dir, fname)
        if not os.path.exists(mask_path):
            print(f"[WARN] Missing mask for {fname}, skipping.")
            continue

        out_path = os.path.join(out_dir, fname)
        make_overlay(img_path, mask_path, out_path)

    print("Overlays saved to:", out_dir)

def main():
    # Input segmentation predictions from the model
    PRED_MASK_DIR = "results/segmentation/test_outputs/masks"
    ORIGINAL_IMG_DIR = "results/segmentation/test_outputs/images"

    OUT_DIR = "results/segmentation/overlays"

    if not os.path.isdir(PRED_MASK_DIR) or not os.path.isdir(ORIGINAL_IMG_DIR):
        raise FileNotFoundError("Predictions not found. Run test_segmentation.py first.")

    process_folder(ORIGINAL_IMG_DIR, PRED_MASK_DIR, OUT_DIR)

if __name__ == "__main__":
    main()
