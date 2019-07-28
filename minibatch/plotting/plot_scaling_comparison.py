import argparse
import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .. import ROOT_DIR

sns.set()


def main(args):
    n_outputs = [10, 100, 1000]

    file_names = []
    labels = []
    linestyles = []

    if args.gpu:
        for n_output in n_outputs:
            file_names.append(
                f"{args.n_input}_input_{n_output}_output_vary_batch_gpu.csv"
            )
            labels.append(f"{n_output} neurons")
            linestyles.append("-")
            file_names.append(
                f"{args.n_input}_input_{n_output}_output_vary_batch_gpu_stdp.csv"
            )
            labels.append(f"{n_output} neurons + STDP")
            linestyles.append("--")
    else:
        for n_output in n_outputs:
            file_names.append(
                f"{args.n_input}_input_{n_output}_output_vary_batch_cpu.csv"
            )
            labels.append(f"{n_output} neurons")
            linestyles.append("-")
            file_names.append(
                f"{args.n_input}_input_{n_output}_output_vary_batch_cpu_stdp.csv"
            )
            labels.append(f"{n_output} neurons + STDP")
            linestyles.append("--")

    colors = ["b", "b", "r", "r", "g", "g"]

    for label, linestyle, color, file_name in zip(
        labels, linestyles, colors, file_names
    ):
        results_file = os.path.join(ROOT_DIR, "results", "scaling", file_name)
        results_path = os.path.join(ROOT_DIR, "results", "scaling", results_file)

        df = pd.read_csv(results_path)
        df = df.groupby("batch size")["simulation time"].agg(["mean", "std"])

        mean = df["mean"].values
        std = df["std"].values

        x = [2 ** i for i in range(len(mean))]
        plt.semilogx(x, mean, label=label, linestyle=linestyle, color=color, marker="+")
        plt.fill_between(x=x, y1=mean - std, y2=mean + std, alpha=0.25, color=color)
        plt.xticks(x, x)

    plt.title(f"Scaling comparison")

    plt.ylim([0.25, 2])
    plt.yticks(np.linspace(0.25, 2, 8), np.linspace(0.25, 2, 8))
    plt.xlabel("Batch size")
    plt.ylabel("Wall clock time (s)")
    plt.ioff()
    plt.legend(prop={"size": 10})
    plt.tight_layout()

    path = os.path.join(ROOT_DIR, "figures", "scaling")
    if not os.path.isdir(path):
        os.makedirs(path)

    if args.gpu:
        file_name = f"{args.n_input}_input_vary_output_vary_batch_vary_learning_gpu.png"
    else:
        file_name = f"{args.n_input}_input_vary_output_vary_batch_vary_learning_cpu.png"

    plt.savefig(os.path.join(path, file_name))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-input", type=int, default=100)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
