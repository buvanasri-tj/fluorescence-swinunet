import os

ROOT = "data"   # folder inside your repo

splits = ["train", "val", "test"]
colors = ["green", "red", "yellow"]

out_train = open("datasets/train_list.txt", "w")
out_val = open("datasets/val_list.txt", "w")
out_test = open("datasets/test_list.txt", "w")

for color in colors:
    for split in splits:
        img_dir = f"{ROOT}/{color}/{split}/images"
        mask_dir = f"{ROOT}/{color}/{split}/masks"

        if split == "test":
            # test images do NOT have masks
            if not os.path.exists(img_dir):
                continue
            for img in sorted(os.listdir(img_dir)):
                if img.endswith(".png"):
                    out_test.write(f"{img_dir}/{img}\n")
            continue

        # For train/val — must include masks
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            continue

        img_files = sorted(os.listdir(img_dir))

        for img in img_files:
            if not img.endswith(".png"):
                continue
            mask = img  # same name as image
            line = f"{img_dir}/{img} {mask_dir}/{mask}\n"
            if split == "train":
                out_train.write(line)
            elif split == "val":
                out_val.write(line)

out_train.close()
out_val.close()
out_test.close()

print("Finished! train_list.txt, val_list.txt, test_list.txt created.")
