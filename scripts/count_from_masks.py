import os
import argparse
import numpy as np
from PIL import Image
import cv2
import csv

def count_objects(mask_path):
    """
    mask_path: path to a binary mask (0 or 255)
    Returns: number of connected components (objects)
    """
    mask = np.array(Image.open(mask_path).convert("L"))
    _, bw = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Connected components
    num_labels, labels = cv2.connectedComponents(bw)

    # Subtract 1 (background)
    return num_labels - 1


def process_folder(mask_dir, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "object_count"])

        for fname in files:
            path = os.path.join(mask_dir, fname)
            count = count_objects(path)
            writer.writerow([fname, count])
            print(f"{fname}: {count} objects")

    print(f"\n[INFO] Saved CSV → {out_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mask_root", type=str, required=True,
                   help="folder containing predicted masks for ONE color")
    p.add_argument("--output_csv", type=str, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_folder(args.mask_root, args.output_csv)
