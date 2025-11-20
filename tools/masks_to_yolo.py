import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

ROOT = "data"
CHANNEL = "green"          # use masks from green channel
SPLITS = ["train", "val", "test"]
CLASS_ID = 0

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_split(split):
    mask_dir = os.path.join(ROOT, CHANNEL, split, "masks")
    out_dir = os.path.join(ROOT, "detection_labels", split)
    ensure_dir(out_dir)

    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    for m in tqdm(mask_files, desc=f"Creating labels for {split}"):
        id_ = os.path.splitext(os.path.basename(m))[0]
        img = cv2.imread(m, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue
        
        # threshold mask
        _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # find connected components
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = bw.shape

        lines = []
        for cnt in contours:
            x, y, bw_w, bw_h = cv2.boundingRect(cnt)

            xc = (x + bw_w/2) / w
            yc = (y + bw_h/2) / h
            wn = bw_w / w
            hn = bw_h / h

            # ignore tiny noise
            if wn < 0.002 or hn < 0.002:
                continue

            lines.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        # write output file
        out_file = os.path.join(out_dir, f"{id_}.txt")
        with open(out_file, "w") as f:
            f.write("\n".join(lines))

def main():
    ensure_dir(os.path.join(ROOT, "detection_labels"))
    for split in SPLITS:
        process_split(split)

if __name__ == "__main__":
    main()
