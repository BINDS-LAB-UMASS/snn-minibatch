import os
import argparse
from time import time

import torch
import bindsnet
import matplotlib.pyplot as plt

from bindsnet.learning import PostPre
from minibatch import ROOT_DIR


def main(args):
    torch.manual_seed(args.seed)
    torch.random.manual_seed(args.seed)

    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # Create network.
    network = bindsnet.network.Network(dt=1.0, batch_size=args.batch_size)

    # Create network components.
    input_layer = bindsnet.network.nodes.Input(n=args.n_input, traces=args.learning)
    output_layer = bindsnet.network.nodes.LIFNodes(
        n=args.n_output, traces=args.learning
    )

    if args.learning:
        connection = bindsnet.network.topology.Connection(
            source=input_layer,
            target=output_layer,
            w=0.1 + 0.1 * torch.randn(args.n_input, args.n_output),
            update_rule=PostPre,
            nu=(1e-4, 1e-2),
        )
    else:
        connection = bindsnet.network.topology.Connection(
            source=input_layer,
            target=output_layer,
            w=0.1 + 0.1 * torch.randn(args.n_input, args.n_output),
        )

    # Add components to network.
    t0 = time()
    network.add_layer(layer=input_layer, name="I")
    network.add_layer(layer=output_layer, name="O")
    network.add_connection(connection=connection, source="I", target="O")

    if args.plot:
        network.add_monitor(
            monitor=bindsnet.network.monitors.Monitor(
                obj=network.layers["O"], state_vars=("s",), time=args.time
            ),
            name="O",
        )

    t1 = time() - t0

    if args.gpu:
        network.to("cuda")

    # Generate random Poisson spike trains with firing rates in [0Hz, 120Hz].
    datum = 120 * torch.rand(args.batch_size, args.n_input)
    input_spikes = bindsnet.encoding.poisson(datum=datum, time=args.time)
    inpts = {"I": input_spikes}
    t2 = time() - t1 - t0

    # Simulate network on input spikes.
    network.run(inpts=inpts, time=args.time)
    t3 = time() - t2 - t1 - t0

    # Report simulation wall-clock time.
    print(
        f"Simulation time: {args.time / 1000} second(s) at 1ms resolution.\n"
        f"Time to initialize network (s): {t1:.2f}\n"
        f"Time to generate data (s): {t2:.2f}\n"
        f"Simulation wall-clock time (s): {t3:.2f}.\n"
    )

    path = os.path.join(ROOT_DIR, "results", args.results_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        with open(os.path.join(path), "w") as f:
            f.write(
                "seed,input neurons,output neurons,time,batch size,init time,"
                "data generation time,simulation time\n"
            )

    to_write = [
        args.seed,
        args.n_input,
        args.n_output,
        args.time,
        args.batch_size,
        t1,
        t2,
        t3,
    ]
    to_write = ",".join(map(str, to_write)) + "\n"
    with open(os.path.join(path), "a") as f:
        f.write(to_write)

    if args.plot:
        # Plot output spikes.
        plt.ioff()
        output_spikes = network.monitors["O"].get("s")[:, 0]
        bindsnet.analysis.plotting.plot_spikes(
            spikes={"I": input_spikes[:, 0], "O": output_spikes}
        )

        from scipy.ndimage import gaussian_filter
        import numpy as np

        output_spikes = output_spikes.cpu().numpy().astype(np.float32)
        output_firing_rates = gaussian_filter(input=output_spikes, sigma=5.0)

        plt.figure()
        for i in range(10):
            plt.plot(output_firing_rates[:, i])
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-file", required=True, type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-input", type=int, default=100)
    parser.add_argument("--n-output", type=int, default=100)
    parser.add_argument("--time", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
