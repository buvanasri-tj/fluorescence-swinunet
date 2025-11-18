import pandas as pd
import matplotlib.pyplot as plt

def plot_curves(csv_path, save_path="results/loss_dice_plot.png"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10,6))
    plt.plot(df["epoch"], df["loss"], label="Loss")
    plt.plot(df["epoch"], df["dice"], label="Dice Score")

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Loss & Dice Curve")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    print("Curve saved to:", save_path)

