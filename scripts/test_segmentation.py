import argparse
import yaml
import torch
from datasets.dataset_fluo import FluoDataset
from torch.utils.data import DataLoader
from models.swinunet import SwinUNet
from utils.postprocess import binarize_mask
from PIL import Image
import os

def parse_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()

    cfg = parse_config(args.cfg)
    img_size = cfg["img_size"]
    test_list = cfg["test_list"]
    outdir = cfg["test_output"]
    threshold = cfg.get("threshold", 0.5)

    os.makedirs(outdir, exist_ok=True)

    print("Loading model...")
    model = SwinUNet(num_classes=1).cuda()
    model.load_state_dict(torch.load(args.weights))

    print("Loading test dataset...")
    ds = FluoDataset(test_list, img_size, mask_required=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    for i, (img, path) in enumerate(dl):
        img = img.cuda()

        with torch.no_grad():
            pred = torch.sigmoid(model(img))

        mask = binarize_mask(pred[0, 0].cpu(), threshold)

        save_path = os.path.join(outdir, os.path.basename(path[0]))
        Image.fromarray(mask).save(save_path)

        print(f"Saved {save_path}")

    print("Inference complete.")
