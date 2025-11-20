import numpy as np
import cv2


def threshold_mask(prob_map, thresh=0.5):
    """
    prob_map: numpy array [0,1]
    returns binary mask
    """
    return (prob_map > thresh).astype("uint8") * 255


def contours_from_mask(mask):
    """
    mask: binary mask (0 or 255)
    returns list of contours
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def bbox_from_contour(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h


def overlay_mask(image, mask):
    """
    image: grayscale or RGB
    mask: binary mask (0/255)
    returns overlay image
    """
    if len(image.shape) == 2:
        image = np.stack([image]*3, axis=-1)

    overlay = image.copy()
    overlay[..., 0] = np.maximum(overlay[..., 0], mask)

    return overlay
