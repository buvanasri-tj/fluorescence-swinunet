import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# --- Fix Python path so imports work ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# --- Repo imports ---
from config import get_config
from models.swinunet import SwinUNet
from datasets.dataset_seg import SegmentationDataset
from scripts.trainer import train_one_epoch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--train_list", type=str, default="datasets/train_list.txt")
    parser.add_argument("--val_list", type=str, default="datasets/val_list.txt")
    parser.add_argument("--output_dir", type=str, default="results/checkpoints")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cfg = get_config(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load model ---
    print("\nLoading SwinUNet model...")
    model = SwinUNet(in_channels=cfg["model"]["in_channels"],
                     num_classes=cfg["model"]["num_classes"]).to(device)

    # --- Load pretrained checkpoint if provided ---
    if "pretrained_ckpt" in cfg["model"]:
        ckpt_path = cfg["model"]["pretrained_ckpt"]
        if os.path.exists(ckpt_path):
            print(f"Loading pretrained checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state, strict=False)
        else:
            print(f"WARNING: Pretrained checkpoint not found: {ckpt_path}")

    # --- Load dataset ---
    print("\nLoading dataset...")
    train_ds = SegmentationDataset(cfg["dataset"]["root_dir"],
                                   split="train",
                                   image_size=cfg["dataset"]["image_size"])

    train_loader = DataLoader(train_ds,
                              batch_size=cfg["training"]["batch_size"],
                              shuffle=True,
                              num_workers=args.num_workers)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])

    # --- Training ---
    print("\nTraining started...")
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(cfg["training"]["epochs"]):
        loss, dice = train_one_epoch(model, train_loader, optimizer, epoch, args.output_dir)
        print(f"Epoch {epoch} Completed | Loss={loss:.4f} | Dice={dice:.4f}")

    print("\nTraining Finished.")
