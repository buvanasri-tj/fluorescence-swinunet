# scripts/train_segmentation.py
# thin wrapper that re-uses your existing trainer.py / train.py logic

import argparse
from trainer import train_one_epoch, validate  # adapt imports to your trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swin_unet.yaml")
    args = parser.parse_args()
    # You already have train.py; keep using it, or call into its functions here.
    # For now: just run existing train.py
    import subprocess
    subprocess.run(["python", "scripts/train.py", "--cfg", args.config], check=True)

if __name__ == "__main__":
    main()
