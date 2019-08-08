import os
import argparse

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from minibatch import ROOT_DIR
from minibatch.conversion.convert import main as convert

sns.set()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--times", type=int, default=None, nargs="+")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, default=None, nargs="+")
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--one-step", action="store_true")
    parser.add_argument("--skip-to-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.times is None:
        args.times = [1, 2, 3, 4, 5, 10]

    if args.batch_sizes is None:
        args.batch_sizes = [2 ** i for i in range(11)]

    if args.gpu:
        args.results_file = os.path.join("conversion", f"vary_time_batch_size_gpu.csv")
    else:
        args.results_file = os.path.join("conversion", f"vary_time_batch_size_cpu.csv")

    results_path = os.path.join(ROOT_DIR, "results", args.results_file)
    if not args.skip_to_plot:
        if os.path.isfile(results_path):
            os.remove(results_path)

        seeds = range(args.n_seeds)

        for seed in seeds:
            for batch_size in args.batch_sizes:
                for time in args.times:
                    print("Random seed:", seed)
                    print("Batch size:", batch_size)
                    print("Simulation time:", time)
                    print()

                    args.seed = seed
                    args.batch_size = batch_size
                    args.time = time

                    try:
                        simulation_time = convert(args)

                        # Disallow simulations longer than 5 minutes.
                        if simulation_time > 5 * 60:
                            break

                    # Simulation causes out-of-memory error.
                    except RuntimeError:
                        break

    df = pd.read_csv(results_path)
    df = df.groupby(["batch size", "simulation time"])[
        "inference time", "accuracy"
    ].agg(["mean", "std"])

    batch_size_mapping = {}
    for row in df.iterrows():
        batch_size = row[0][0]
        sim_time = row[0][1]
        inf_time_mean = row[1]["inference time"]["mean"]
        inf_time_std = row[1]["inference time"]["std"]
        accuracy = row[1]["accuracy"]["mean"]

        stats = (inf_time_mean, inf_time_std, accuracy)
        if batch_size not in batch_size_mapping:
            batch_size_mapping[batch_size] = {sim_time: stats}
        else:
            batch_size_mapping[batch_size][sim_time] = stats

    with sns.cubehelix_palette(len(batch_size_mapping), start=0.5, rot=-0.75):
        fig, ax = plt.subplots()
        for batch_size in batch_size_mapping:
            sim_time_mapping = batch_size_mapping[batch_size]

            x = sorted(sim_time_mapping.keys())
            y = sorted([stats[0] for stats in sim_time_mapping.values()])
            y_stdev = sorted(
                1 * x
                for (_, x) in zip(
                    [stats[0] for stats in sim_time_mapping.values()],
                    [stats[1] for stats in sim_time_mapping.values()],
                )
            )
            y = np.array(y)
            y_stdev = np.array(y_stdev)
            ax.loglog(x, y, label=batch_size)
            # ax.fill_between(x, y1=y - y_stdev, y2=y + y_stdev, alpha=0.25)

            if batch_size == 128:
                save_x = x

        plt.ioff()
        plt.legend(loc=2, fontsize=9)
        plt.xlabel("Simulated time (ms)")
        plt.ylabel("Wall-clock time (s)")
        plt.title("Test set inference time: simulated vs. wall-clock")
        plt.xticks(save_x, save_x)

    path = os.path.join(ROOT_DIR, "figures", "conversion")
    if not os.path.isdir(path):
        os.makedirs(path)

    if args.gpu:
        file_name = f"inf_time_convert_vary_time_batch_size_gpu.png"
    else:
        file_name = f"inf_time_convert_vary_time_batch_size_cpu.png"

    plt.savefig(os.path.join(path, file_name))

    with sns.cubehelix_palette(len(batch_size_mapping), start=0.5, rot=-0.75):
        fig, ax = plt.subplots()
        for batch_size in batch_size_mapping:
            sim_time_mapping = batch_size_mapping[batch_size]

            x = sorted(sim_time_mapping.keys())
            y = sorted(
                [
                    stats[0] / (10000 // batch_size)
                    for stats in sim_time_mapping.values()
                ]
            )
            y_stdev = sorted(
                1 * x
                for (_, x) in zip(
                    [
                        stats[0] / (10000 // batch_size)
                        for stats in sim_time_mapping.values()
                    ],
                    [
                        stats[1] / (10000 // batch_size)
                        for stats in sim_time_mapping.values()
                    ],
                )
            )
            y = np.array(y)
            y_stdev = np.array(y_stdev)
            ax.loglog(x, y, label=batch_size)
            # ax.fill_between(x, y1=y - y_stdev, y2=y + y_stdev, alpha=0.25)

            if batch_size == 128:
                save_x = x

        plt.ioff()
        plt.legend(loc="lower right", fontsize=9)
        plt.xlabel("Simulated time (ms)")
        plt.ylabel("Wall-clock time per batch (s)")
        plt.title("Time per minibatch: simulated vs. wall-clock")
        plt.xticks(save_x, save_x)

    if args.gpu:
        file_name = f"steps_per_second_convert_vary_time_batch_size_gpu.png"
    else:
        file_name = f"steps_per_second_convert_vary_time_batch_size_cpu.png"

    plt.savefig(os.path.join(path, file_name))

    mapping128 = batch_size_mapping[128]

    fig, ax = plt.subplots()
    x = sorted(mapping128.keys())
    y = sorted([stats[2] for stats in mapping128.values()])
    plt.semilogx(x, y, label="SNN")
    plt.axhline(98.31, color="r", linestyle=":", label="ANN")
    plt.xlabel("Simulated time (ms)")
    plt.ylabel("Test set accuracy")
    plt.title("Converted SNN accuracy")
    plt.xticks(x, x)
    plt.legend()

    if args.gpu:
        file_name = f"accuracy_convert_vary_time_batch_size_gpu.png"
    else:
        file_name = f"accuracy_convert_vary_time_batch_size_cpu.png"

    plt.savefig(os.path.join(path, file_name))
    plt.show()
