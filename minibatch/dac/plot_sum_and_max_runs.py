import os

import pandas as pd
import seaborn as sns
from scipy import signal
import matplotlib.pyplot as plt

from minibatch import ROOT_DIR


def main():
    sns.set()

    # Sum reduction runs.
    # Accuracy by no. of training examples.
    path = os.path.join(ROOT_DIR, "data", "dac", "sum")
    files = os.listdir(path)
    dfs = []
    batch_sizes = []
    for f in files:
        dfs.append(pd.read_csv(os.path.join(path, f)))
        batch_sizes.append(int(f.split('-')[2]))

    dfs = [df for _, df in sorted(zip(batch_sizes, dfs))]
    batch_sizes = sorted(batch_sizes)

    with sns.cubehelix_palette(len(batch_sizes), start=0.5, rot=-0.75):
        fig, ax = plt.subplots()
        for batch_size, df in zip(batch_sizes, dfs):
            step = df["Step"]  # * max(batch_size, 250)
            win = signal.hann(10)
            filtered = signal.convolve(df["Value"], win, mode='same') / sum(win)
            step = step[:-10]
            filtered = 100 * filtered[:-10]
            ax.plot(step, filtered, label=batch_size)

    plt.xlabel("No. of training examples (1 epoch)")
    plt.xticks(range(0, 70000, 10000), ["0K", "10K", "20K", "30K", "40K", "50K", "60K"])
    plt.ylabel("Test accuracy")
    plt.title("Accuracy comparison: average parameter reduction")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(
        ROOT_DIR, "figures", "dac", "avg_reduction_training_examples.png"
    )

    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    plt.savefig(path)
    plt.show()

    # Max reduction runs.
    # Accuracy by no. of training examples.
    path = os.path.join(ROOT_DIR, "data", "dac", "max")
    files = os.listdir(path)
    dfs = []
    batch_sizes = []
    for f in files:
        dfs.append(pd.read_csv(os.path.join(path, f)))
        batch_sizes.append(int(f.split('-')[2]))

    dfs = [df for _, df in sorted(zip(batch_sizes, dfs))]
    batch_sizes = sorted(batch_sizes)

    with sns.cubehelix_palette(len(batch_sizes), start=0.5, rot=-0.75):
        fig, ax = plt.subplots()
        for batch_size, df in zip(batch_sizes, dfs):
            step = df["Step"]  # * max(batch_size, 250)
            win = signal.hann(10)
            filtered = signal.convolve(df["Value"], win, mode='same') / sum(win)
            step = step[:-10]
            filtered = 100 * filtered[:-10]
            ax.plot(step, filtered, label=batch_size)

    plt.xlabel("No. of training examples (1 epoch)")
    plt.xticks(range(0, 70000, 10000), ["0K", "10K", "20K", "30K", "40K", "50K", "60K"])
    plt.ylabel("Test accuracy")
    plt.title("Accuracy comparison: maximum parameter reduction")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(
        ROOT_DIR, "figures", "dac", "max_reduction_training_examples.png"
    )

    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    plt.savefig(path)
    plt.show()

    # Accuracy by wall times.
    with sns.cubehelix_palette(len(batch_sizes), start=0.5, rot=-0.75):
        fig, ax = plt.subplots()
        for batch_size, df in zip(batch_sizes[::-1], dfs[::-1]):
            time = (df["Wall time"] - df["Wall time"].min()) / 60
            win = signal.hann(10)
            filtered = signal.convolve(df["Value"], win, mode='same') / sum(win)
            time = time[:-10]
            filtered = 100 * filtered[:-10]
            ax.semilogx(time, filtered, label=batch_size)

    plt.xlim([1e-1, None])
    plt.ylim([50, 90])
    plt.xlabel("Wall clock time (m) (1 epoch)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs. wall-clock time")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(ROOT_DIR, "figures", "dac", "max_reduction_wall_clock_time.png")

    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    plt.savefig(path)
    plt.show()


if __name__ == "__main__":
    main()
