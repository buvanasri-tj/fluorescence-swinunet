# utils/postprocess.py
import numpy as np, cv2
def mask_to_instances(mask, threshold=0.5):
    if mask.dtype != np.uint8:
        mask = (mask > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(c) for c in contours]
    return bboxes, contours
