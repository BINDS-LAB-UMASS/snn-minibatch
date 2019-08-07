import os
import argparse
from time import time

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

from bindsnet.analysis.plotting import plot_spikes
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import Hebbian
from minibatch import ROOT_DIR


def main(args):
    # Random seeding.
    torch.manual_seed(args.seed)

    # Device.
    device = torch.device("cuda" if args.gpu else "cpu")

    # Build network.
    network = Network(batch_size=args.batch_size)
    network.add_layer(
        Input(shape=(1, 28, 28), traces=True), name="I"
    )
    network.add_layer(
        LIFNodes(n=10, traces=True, rest=0, reset=0, thresh=1), name="O"
    )
    network.add_connection(
        Connection(
            source=network.layers["I"],
            target=network.layers["O"],
            nu=(0.01, 0.01),
            update_rule=Hebbian,
        ),
        source="I",
        target="O",
    )

    if args.plot:
        for l in network.layers:
            network.add_monitor(
                Monitor(network.layers[l], state_vars=("s",), time=args.time), name=l
            )

    network.to(device)

    # Load dataset.
    dataset = MNIST(
        image_encoder=PoissonEncoder(time=args.time, dt=1.0),
        label_encoder=None,
        root=os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 250),
            ]
        ),
    )

    # Create a dataloader to iterate and batch data
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=args.gpu,
    )

    spike_ims = None
    spike_axes = None

    t0 = time()
    for step, batch in enumerate(tqdm(dataloader)):
        # Prep next input batch.
        inputs = batch["encoded_image"]

        inpts = {"Input": inputs}
        if args.gpu:
            inpts = {k: v.cuda() for k, v in inpts.items()}

        # Run the network on the input.
        network.run(inpts=inpts, time=args.time, one_step=args.one_step)

        if args.plot:
            # Plot output spikes.
            spikes = {
                l: network.monitors[l].get("s")[:, 0] for l in network.monitors
            }
            spike_ims, spike_axes = plot_spikes(
                spikes=spikes, ims=spike_ims, axes=spike_axes
            )
            plt.pause(1e-4)

        # Reset state variables.
        network.reset_()

    t1 = time() - t0

    print(f"Time: {t1}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--time", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--one-step", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())