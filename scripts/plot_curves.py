import os
import matplotlib.pyplot as plt

def parse_log(log_path):
    """Extract epoch, loss, and dice values from training log."""
    epochs = []
    losses = []
    dices = []

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Epoch") and "Loss=" in line:
                # Example: "Epoch 4/50 | Loss=0.9283 | Dice=0.1251"
                parts = line.split("|")
                epoch_info = parts[0].strip().split()[1]      # "4/50"
                loss_info = parts[1].strip().split("=")[1]    # "0.9283"
                dice_info = parts[2].strip().split("=")[1]    # "0.1251"

                epoch_num = int(epoch_info.split("/")[0])
                epochs.append(epoch_num)
                losses.append(float(loss_info))
                dices.append(float(dice_info))

    return epochs, losses, dices


def plot_curves(epochs, losses, dices, out_dir="results/segmentation/plots"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Loss Curve
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Swin-UNet Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curve.png"))
    plt.close()

    # Dice Curve
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, dices, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Swin-UNet Dice Score Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_curve.png"))
    plt.close()

    print("Saved curves to:", out_dir)


def main():
    LOG_FILE = "results/segmentation/train_log.txt"

    if not os.path.exists(LOG_FILE):
        raise FileNotFoundError(f"Training log not found: {LOG_FILE}")

    epochs, losses, dices = parse_log(LOG_FILE)
    plot_curves(epochs, losses, dices)


if __name__ == "__main__":
    main()
