import argparse
import os
import torch
from torch.utils.data import DataLoader

import sys

# Add repository root to PYTHONPATH
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from config import get_config
from networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
from datasets.dataset_fluo import FluoDataset
from trainer import train_one_epoch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--train_list", type=str, default="datasets/train_list.txt")
    parser.add_argument("--val_list", type=str, default="datasets/test_list.txt")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = get_config(args)

    print("\nLoading model...")
    model = SwinUnet(config, img_size=args.img_size, num_classes=2).cuda()
    model.load_from(config)

    print("\nLoading training dataset...")
    train_ds = FluoDataset(args.train_list, img_size=args.img_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.TRAIN.BASE_LR)

    print("\nTraining started...")
    for epoch in range(args.num_epochs):
        loss, dice = train_one_epoch(model, train_loader, optimizer, epoch, args.output_dir)
        print(f"Epoch {epoch} Completed | Loss={loss:.4f} | Dice={dice:.4f}")

    print("\nTraining Finished.")
