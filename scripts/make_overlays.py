import os
import numpy as np
from PIL import Image


def overlay(img_path, mask_path, save_path):
    img = np.array(Image.open(img_path).convert("L"))
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask > 127).astype(np.uint8) * 255

    color = np.stack([img, img, img], axis=-1)
    color[:, :, 0] = np.maximum(color[:, :, 0], mask)

    Image.fromarray(color).save(save_path)


def main():
    os.makedirs("overlays", exist_ok=True)
    imgs = sorted(os.listdir("predictions"))
    for img in imgs:
        overlay(f"data/green/test/images/{img}",
                f"predictions/{img}",
                f"overlays/{img}")


if __name__ == "__main__":
    main()
