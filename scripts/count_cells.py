import argparse
import torch
from models.swinunet import SwinUNet
from utils.postprocess import mask_to_centroids
from datasets.dataset_fluo import FluoDataset
from torch.utils.data import DataLoader
import yaml

def load_cfg(path):
    import yaml
    return yaml.safe_load(open(path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    img_size = cfg["img_size"]
    threshold = cfg["threshold"]
    test_list = cfg["test_list"]

    print("Loading model...")
    model = SwinUNet(num_classes=1).cuda()
    model.load_state_dict(torch.load(args.weights))

    ds = FluoDataset(test_list, img_size, mask_required=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    print("Counting cells...")
    for img, path in dl:
        img = img.cuda()

        with torch.no_grad():
            pred = torch.sigmoid(model(img))

        centroids = mask_to_centroids(pred[0,0].cpu(), threshold)
        print(path[0], "→", len(centroids), "cells detected")
