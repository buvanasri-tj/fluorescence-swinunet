import os
import numpy as np
from PIL import Image

def make_overlay(image_path, mask_path, output_path, color=(255,0,0), alpha=0.4):
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask)

    overlay = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    overlay[mask_np > 0] = color

    overlay_img = Image.fromarray(overlay)
    blended = Image.blend(img, overlay_img, alpha)

    blended.save(output_path)


def batch_overlay(images_dir, preds_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    images = sorted(os.listdir(images_dir))
    preds = sorted(os.listdir(preds_dir))

    for img_name, pred_name in zip(images, preds):
        make_overlay(
            os.path.join(images_dir, img_name),
            os.path.join(preds_dir, pred_name),
            os.path.join(output_dir, img_name)
        )

    print(f"Overlay images saved to: {output_dir}")
