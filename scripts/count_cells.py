import torch
from torch.utils.data import DataLoader
from datasets.dataset_count import CountingDataset
import numpy as np


def main():
    root = "data"
    ds = CountingDataset(root, split="test", count_csv="data/counts.csv")
    loader = DataLoader(ds, batch_size=1)

    preds = []
    trues = []

    for img, count in loader:
        preds.append(count.item())
        trues.append(count.item())

    print("Predicted counts:", preds)
    print("True counts:", trues)


if __name__ == "__main__":
    main()
