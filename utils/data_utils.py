import glob

def create_list_file(images_dir, masks_dir, output_txt):
    images = sorted(glob.glob(images_dir + "/*.png"))
    masks  = sorted(glob.glob(masks_dir + "/*.png"))

    with open(output_txt, "w") as f:
        for img, mask in zip(images, masks):
            f.write(f"{img} {mask}\n")
