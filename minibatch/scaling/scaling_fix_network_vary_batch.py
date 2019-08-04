import os
import argparse

import torch
import cpuinfo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from minibatch import ROOT_DIR
from minibatch.scaling.scaling import main as scaling

sns.set()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-input", type=int, default=100)
    parser.add_argument("--n-output", type=int, default=100)
    parser.add_argument("--time", type=int, default=1000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--power-of-two", type=int, default=10)
    parser.add_argument("--learning", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--skip-to-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.gpu:
        if args.learning:
            args.results_file = os.path.join(
                "scaling",
                f"{args.n_input}_input_{args.n_output}_output_vary_batch_gpu_stdp.csv",
            )
        else:
            args.results_file = os.path.join(
                "scaling",
                f"{args.n_input}_input_{args.n_output}_output_vary_batch_gpu.csv",
            )
    else:
        if args.learning:
            args.results_file = os.path.join(
                "scaling",
                f"{args.n_input}_input_{args.n_output}_output_vary_batch_cpu_stdp.csv",
            )
        else:
            args.results_file = os.path.join(
                "scaling",
                f"{args.n_input}_input_{args.n_output}_output_vary_batch_cpu.csv",
            )

    results_path = os.path.join(ROOT_DIR, "results", args.results_file)
    if not args.skip_to_plot:
        if os.path.isfile(results_path):
            os.remove(results_path)

        seeds = range(args.n_seeds)
        batch_sizes = [2 ** i for i in range(args.power_of_two + 1)]

        for seed in seeds:
            for batch_size in batch_sizes:
                args.seed = seed
                args.batch_size = batch_size

                scaling(args)

    df = pd.read_csv(results_path)
    df = df.groupby("batch size")["simulation time"].agg(["mean", "std"])

    mean = df["mean"].values
    std = df["std"].values

    x = [2 ** i for i in range(len(mean))]
    plt.semilogx(x, mean)
    plt.fill_between(x=x, y1=mean - std, y2=mean + std, alpha=0.25)
    plt.xticks(x, x)

    if args.gpu:
        device = torch.cuda.get_device_name(0)
        plt.title(
            f"{args.n_input} input, " f"{args.n_output} output neurons " f"({device})",
            fontdict={"size": 12},
        )
    else:
        device = cpuinfo.get_cpu_info()["brand"]
        plt.title(
            f"{args.n_input} input, " f"{args.n_output} output neurons " f"({device})",
            fontdict={"size": 10},
        )

    plt.xlabel("Batch size")
    plt.ylabel("Wall clock time (s)")
    plt.ioff()

    path = os.path.join(ROOT_DIR, "figures", "scaling")
    if not os.path.isdir(path):
        os.makedirs(path)

    if args.gpu:
        if args.learning:
            file_name = (
                f"{args.n_input}_input_{args.n_output}_output_vary_batch_gpu_stdp.png"
            )
        else:
            file_name = (
                f"{args.n_input}_input_{args.n_output}_output_vary_batch_gpu.png"
            )
    else:
        if args.learning:
            file_name = (
                f"{args.n_input}_input_{args.n_output}_output_vary_batch_csv_stdp.png"
            )
        else:
            file_name = (
                f"{args.n_input}_input_{args.n_output}_output_vary_batch_csv.png"
            )

    plt.savefig(os.path.join(path, file_name))
    plt.show()
