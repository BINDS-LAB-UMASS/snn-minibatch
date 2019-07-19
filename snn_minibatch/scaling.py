import argparse
from time import time

import torch
import bindsnet
import matplotlib.pyplot as plt


def main(args):
    # Random seeding.
    torch.random.manual_seed(args.seed)

    # Create network.
    network = bindsnet.network.Network(batch_size=args.batch_size)

    # Create network components.
    input_layer = bindsnet.network.nodes.Input(n=args.n_input)
    output_layer = bindsnet.network.nodes.LIFNodes(n=args.n_output)
    connection = bindsnet.network.topology.Connection(
        source=input_layer,
        target=output_layer,
        w=0.05 * torch.rand(args.n_input, args.n_output),
    )

    # Add components to network.
    t0 = time()
    network.add_layer(layer=input_layer, name="I")
    network.add_layer(layer=output_layer, name="O")
    network.add_connection(connection=connection, source="I", target="O")
    network.add_monitor(
        monitor=bindsnet.network.monitors.Monitor(
            obj=network.layers["O"], state_vars=("s"), time=args.time
        ),
        name="O",
    )
    t1 = time() - t0

    # Generate random Poisson spike trains with firing rates in [0Hz, 120Hz].
    datum = 120 * torch.rand(args.batch_size, args.n_input)
    input_spikes = bindsnet.encoding.poisson(datum=datum, time=args.time)
    inputs = {"I": input_spikes}
    t2 = time() - t1 - t0

    # Simulate network on input spikes.
    network.run(inputs=inputs, time=args.time)
    t3 = time() - t2 - t1 - t0

    # Report simulation wall-clock time.
    print(
        f"Simulation time: {args.time / 1000} second(s) at {args.dt}ms resolution.\n"
        f"Time to initialize network (s): {t1:.2f}\n"
        f"Time to generate data (s): {t2:.2f}\n"
        f"Simulation wall-clock time (s): {t3:.2f}."
    )

    if args.plot:
        # Plot output spikes.
        plt.ioff()
        output_spikes = network.monitors["O"].get("s")[:, 0]
        bindsnet.analysis.plotting.plot_spikes(
            spikes={"I": input_spikes[:, 0], "O": output_spikes}
        )

        from scipy.ndimage import gaussian_filter
        import numpy as np

        output_spikes = output_spikes.numpy().astype(np.float32)
        output_firing_rates = gaussian_filter(input=output_spikes, sigma=5.0)

        plt.figure()
        for i in range(10):
            plt.plot(output_firing_rates[:, i])
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--n-input", type=int, default=100)
    parser.add_argument("--n-output", type=int, default=100)
    parser.add_argument("--time", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
