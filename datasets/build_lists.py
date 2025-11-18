import os

# Root directory of your dataset
ROOT = "data"

# Output list files
TRAIN_OUT = "datasets/train_list.txt"
TEST_OUT = "datasets/test_list.txt"

# Valid color channels
COLORS = ["green", "red", "yellow"]

def pair_paths(image_dir, mask_dir):
    """
    Returns sorted list of (image_path, mask_path) for matching filenames.
    """
    image_files = sorted(os.listdir(image_dir))

    pairs = []
    for img in image_files:
        img_path = os.path.join(image_dir, img)
        mask_path = os.path.join(mask_dir, img)

        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            print(f"[WARNING] Mask not found for {img_path}")
    return pairs


def build_list(list_path, data_type="train"):
    """
    Writes list of PNG pairs to list_path
    """
    with open(list_path, "w") as f:
        for color in COLORS:
            img_dir = os.path.join(ROOT, color, data_type, "images")
            mask_dir = os.path.join(ROOT, color, data_type, "masks")

            if not os.path.exists(img_dir):
                print(f"[WARNING] Missing folder: {img_dir}")
                continue

            pairs = pair_paths(img_dir, mask_dir)

            for img, mask in pairs:
                f.write(f"{img} {mask}\n")

    print(f"[OK] Created {list_path}")


if __name__ == "__main__":
    print("Building train/val lists...")

    build_list(TRAIN_OUT, data_type="train")
    build_list(TEST_OUT, data_type="val")

    print("Done.")
