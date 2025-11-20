import numpy as np
import cv2

def binarize_mask(mask_tensor, threshold):
    mask = mask_tensor.numpy()
    mask = (mask > threshold).astype(np.uint8) * 255
    return mask

def mask_to_centroids(mask_tensor, threshold=0.5):
    mask = (mask_tensor.numpy() > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])
            centers.append((cx, cy))

    return centers
