# scripts/count_cells.py
import cv2, numpy as np
from skimage import measure
from pathlib import Path
import yaml

def count_from_mask(mask_path, min_area=20):
    mask = cv2.imread(mask_path, 0)
    _, thr = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    labels = measure.label(thr, connectivity=2)
    props = measure.regionprops(labels)
    count = sum(1 for p in props if p.area >= min_area)
    return count

if __name__ == "__main__":
    # Example usage: iterate through results folder and count
    import sys
    mask = sys.argv[1]
    print("count:", count_from_mask(mask))
