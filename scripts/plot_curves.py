import matplotlib.pyplot as plt
import json


def main():
    with open("logs.json") as f:
        logs = json.load(f)

    plt.plot(logs["train"], label="Train")
    plt.plot(logs["val"], label="Val")
    plt.legend()
    plt.savefig("curves.png")


if __name__ == "__main__":
    main()
