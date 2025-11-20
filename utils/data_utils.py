import os
from PIL import Image
import numpy as np


def load_gray(path):
    return Image.open(path).convert("L")


def normalize_img(img):
    """
    img: numpy array
    returns: normalized [0,1] float32
    """
    img = np.array(img).astype("float32")
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    return img


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
