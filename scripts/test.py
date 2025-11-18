import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append('/content/fluorescence-swinunet')

from networks.vision_transformer import SwinUnet
from datasets.dataset_fluo import FluoDataset
from config import get_config
from utils.metrics import dice_score


def save_prediction(mask, output_path):
    mask = (mask * 255).astype(np.uint8)
    Image.fromarray(mask).save(output_path)


def inference(args):
    config = get_config(args)

    print("\nLoading model...")
    model = SwinUnet(config, num_classes=2).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
    model.eval()

    print("\nLoading validation dataset...")
    dataset = FluoDataset(args.val_list, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(args.output_dir, exist_ok=True)

    dice_scores = []

    print("\nRunning inference...")
    for i, (img, mask) in enumerate(tqdm(loader)):
        img = img.cuda()

        with torch.no_grad():
            pred = model(img)
            pred_sig = torch.sigmoid(pred)
            pred_bin = (pred_sig > 0.5).float()

        dice_scores.append(dice_score(pred, mask))

        pred_np = pred_bin.squeeze().cpu().numpy()
        save_prediction(pred_np, f"{args.output_dir}/pred_{i}.png")

    avg_dice = np.mean(dice_scores)
    print(f"\nFinished inference.")
    print(f"Average Dice Score = {avg_dice:.4f}")

    with open(f"{args.output_dir}/metrics.txt", "w") as f:
        f.write(f"Average Dice: {avg_dice:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val_list", type=str, default="datasets/test_list.txt")
    parser.add_argument("--output_dir", type=str, default="results/predictions")
    parser.add_argument("--img_size", type=int, default=224)

    args = parser.parse_args()

    inference(args)
