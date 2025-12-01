import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("counts", exist_ok=True)

colors = ["green", "yellow", "red"]

for color in colors:
    csv_path = f"counts/{color}_counts.csv"

    if not os.path.exists(csv_path):
        print(f"[WARN] Missing: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    # Histogram
    plt.figure(figsize=(6,4))
    plt.hist(df["object_count"], bins=20)
    plt.title(f"{color.upper()} – Object Count Distribution")
    plt.xlabel("Objects per image")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"counts/{color}_hist.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(4,6))
    plt.boxplot(df["object_count"], vert=True)
    plt.title(f"{color.upper()} – Count Variation")
    plt.ylabel("Objects per image")
    plt.grid(True)
    plt.savefig(f"counts/{color}_boxplot.png")
    plt.close()

    print(f"[OK] Saved plots for {color}")
