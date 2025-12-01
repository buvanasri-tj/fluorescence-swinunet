import pandas as pd
import matplotlib.pyplot as plt

colors = ["green", "yellow", "red"]

for color in colors:
    df = pd.read_csv(f"counts/{color}_counts.csv")

    plt.figure(figsize=(6,4))
    plt.hist(df["object_count"], bins=20)
    plt.title(f"{color.upper()} – Object Count Distribution")
    plt.xlabel("Count per image")
    plt.ylabel("Frequency")
    plt.savefig(f"counts/{color}_hist.png")
    plt.close()

    plt.figure(figsize=(4,6))
    plt.boxplot(df["object_count"])
    plt.title(f"{color.upper()} – Count Variation")
    plt.ylabel("Objects per image")
    plt.savefig(f"counts/{color}_boxplot.png")
    plt.close()
