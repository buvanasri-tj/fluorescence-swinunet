import csv
import matplotlib.pyplot as plt
import os

log_path = "checkpoints/seg/log.csv"
out_dir = "plots"
os.makedirs(out_dir, exist_ok=True)

epochs = []
train_loss = []
train_dice = []
val_loss = []
val_dice = []

with open(log_path, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        epochs.append(int(r["epoch"]))
        train_loss.append(float(r["train_loss"]))
        train_dice.append(float(r["train_dice"]))
        val_loss.append(float(r["val_loss"]))
        val_dice.append(float(r["val_dice"]))

# Loss curve
plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, val_loss, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{out_dir}/loss_curve.png")
plt.close()

# Dice curve
plt.figure(figsize=(8,5))
plt.plot(epochs, train_dice, label="Train Dice")
plt.plot(epochs, val_dice, label="Val Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.legend()
plt.grid(True)
plt.savefig(f"{out_dir}/dice_curve.png")
plt.close()

print("[INFO] Plots saved in plots/")
