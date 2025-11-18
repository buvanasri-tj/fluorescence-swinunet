import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

import torch
from torch.utils.data import DataLoader

# -------------------------------------------------------------------
# Imports from the repository
# -------------------------------------------------------------------
from config import get_config
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from datasets.dataset_fluo import FluoDataset
from scripts.trainer import train_one_epoch   # correct path


# -------------------------------------------------------------------
# Argument parser
# -------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, required=True,
                        help="Path to YAML config (configs/swin_unet.yaml)")

    parser.add_argument("--train_list", type=str,
                        default="datasets/train_list.txt")

    parser.add_argument("--val_list", type=str,
                        default="datasets/test_list.txt")

    parser.add_argument("--output_dir", type=str,
                        default="results/checkpoints")

    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=2)

    return parser.parse_args()


# -------------------------------------------------------------------
# Main training
# -------------------------------------------------------------------
if __name__ == "__main__":
    args = get_args()
    config = get_config(args)

    # Create output folder
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading Swin-UNet model...")
    model = SwinTransformerSys(
        img_size=args.img_size,
        num_classes=2,
        in_chans=1
    ).cuda()

    print("Model loaded.")

    print("\nLoading training dataset...")
    train_ds = FluoDataset(args.train_list, img_size=args.img_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TRAIN.BASE_LR)

    print("\nTraining started...\n")

    for epoch in range(args.num_epochs):
        loss, dice = train_one_epoch(
            model,
            train_loader,
            optimizer,
            epoch,
            args.output_dir
        )

        print(f"Epoch {epoch+1}/{args.num_epochs} | Loss={loss:.4f} | Dice={dice:.4f}")

    print("\nTraining Completed.")
