import argparse
import os

import torch
import cpuinfo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .. import ROOT_DIR

sns.set()


def main(args):
    if args.gpu:
        file_names = [
            f"{args.n_input}_input_{args.n_output}_output_vary_batch_gpu.csv",
            f"{args.n_input}_input_{args.n_output}_output_vary_batch_gpu_stdp.csv",
        ]
    else:
        file_names = [
            f"{args.n_input}_input_{args.n_output}_output_vary_batch_cpu.csv",
            f"{args.n_input}_input_{args.n_output}_output_vary_batch_cpu_stdp.csv",
        ]

    for i, file_name in enumerate(file_names):
        results_file = os.path.join(ROOT_DIR, "results", "scaling", file_name)
        results_path = os.path.join(ROOT_DIR, "results", "scaling", results_file)

        df = pd.read_csv(results_path)
        df = df.groupby("batch size")["simulation time"].agg(["mean", "std"])

        mean = df["mean"].values
        std = df["std"].values

        x = [2 ** i for i in range(len(mean))]

        if i == 0:
            plt.semilogx(x, mean, label="No STDP")
        elif i == 1:
            plt.semilogx(x, mean, label="STDP")

        plt.fill_between(x=x, y1=mean - std, y2=mean + std, alpha=0.25)
        plt.xticks(x, x)

    if args.gpu:
        device = torch.cuda.get_device_name(0)
        plt.title(
            f"{args.n_input}, {args.n_output} input, output neurons ({device})",
            fontdict={"size": 12},
        )
    else:
        device = cpuinfo.get_cpu_info()["brand"]
        plt.title(
            f"{args.n_input}, {args.n_output} input, output neurons ({device})",
            fontdict={"size": 10},
        )

    plt.xlabel("Batch size")
    plt.ylabel("Wall clock time (s)")
    plt.ioff()
    plt.legend()

    path = os.path.join(ROOT_DIR, "figures", "scaling")
    if not os.path.isdir(path):
        os.makedirs(path)

    if args.gpu:
        file_name = f"{args.n_input}_input_{args.n_output}_output_vary_batch_gpu_vary_learning.png"
    else:
        file_name = f"{args.n_input}_input_{args.n_output}_output_vary_batch_cpu_vary_learning.png"

    plt.savefig(os.path.join(path, file_name))
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-input", type=int, default=100)
    parser.add_argument("--n-output", type=int, default=100)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
