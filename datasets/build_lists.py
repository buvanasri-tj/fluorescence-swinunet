import os
import glob

def gather_images(base):
    """Return all .png images inside images/ folders under base."""
    return sorted(glob.glob(os.path.join(base, "*", "images", "*.png")))

def write_list(path, items):
    with open(path, "w") as f:
        for x in items:
            f.write(x.replace("\\", "/") + "\n")

if __name__ == "__main__":

    # Root dataset folder
    ROOT = "data"

    # Final text list files
    TRAIN_LIST = "datasets/train_list.txt"
    VAL_LIST   = "datasets/val_list.txt"
    TEST_LIST  = "datasets/test_list.txt"

    print("Building train/val/test lists...")

    train_images = gather_images(os.path.join(ROOT, "*", "train"))
    val_images   = gather_images(os.path.join(ROOT, "*", "val"))
    test_images  = gather_images(os.path.join(ROOT, "*", "test"))

    write_list(TRAIN_LIST, train_images)
    write_list(VAL_LIST, val_images)
    write_list(TEST_LIST, test_images)

    print(f"Created {TRAIN_LIST}   -> {len(train_images)} images")
    print(f"Created {VAL_LIST}     -> {len(val_images)} images")
    print(f"Created {TEST_LIST}    -> {len(test_images)} images")
