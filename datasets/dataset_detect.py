# datasets/dataset_detect.py
# Convert segmentation masks to YOLO-style bbox labels (x_center y_center w h, normalized)

import os
import cv2
import numpy as np

def masks_to_bboxes(mask_path):
    mask = cv2.imread(mask_path, 0)
    _, thr = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h,w = mask.shape
    for c in contours:
        x,y,ww,hh = cv2.boundingRect(c)
        x_center = (x + ww/2)/w
        y_center = (y + hh/2)/h
        boxes.append((0, x_center, y_center, ww/w, hh/h))  # class 0
    return boxes

def save_yolo_label(label_path, boxes):
    with open(label_path, 'w') as f:
        for b in boxes:
            f.write(" ".join(map(str, b)) + "\n")
