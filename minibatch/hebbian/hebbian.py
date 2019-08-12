import os
import argparse
from time import time

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

from bindsnet.analysis.plotting import plot_spikes, plot_weights
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.learning import LearningRule, Hebbian
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights
from minibatch import ROOT_DIR


# class Oja(LearningRule):
#
#     def __init__(self, connection, nu=None, reduction=None, weight_decay=0.0, **kwargs):
#         super().__init__(
#             connection=connection,
#             nu=nu,
#             reduction=reduction,
#             weight_decay=weight_decay,
#             **kwargs
#         )
#
#         assert (
#             self.source.traces and self.target.traces
#         ), "Both pre- and post-synaptic nodes must record spike traces."
#
#         self.update = self._connection_update
#
#     def _connection_update(self, **kwargs) -> None:
#         batch_size = self.source.batch_size
#
#         source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
#         target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
#         target_v = self.target.v.view(batch_size, -1).unsqueeze(1)
#
#         # Post-synaptic update.
#         update1 = self.reduction(torch.bmm(source_x, target_s), dim=0)
#         print(target_v.shape, self.connection.w[None].shape)
#         update2 = torch.mean(target_v ** 2 * self.connection.w[None], dim=0)
#         self.connection.w += self.nu[1] * (update1 - update2)
#
#         super().update()


def main(args):
    # Random seeding.
    torch.manual_seed(args.seed)

    # Device.
    device = torch.device("cuda" if args.gpu else "cpu")

    # No. workers.
    if args.n_workers == -1:
        args.n_workers = args.gpu * 4 * torch.cuda.device_count()

    # Build network.
    network = Network(batch_size=args.batch_size)
    network.add_layer(Input(shape=(1, 28, 28), traces=True), name="I")
    network.add_layer(
        LIFNodes(n=10, traces=True, rest=0, reset=0, thresh=1, refrac=0), name="O"
    )
    network.add_connection(
        Connection(
            source=network.layers["I"],
            target=network.layers["O"],
            nu=(0.0, 0.01),
            update_rule=Hebbian,
            wmin=0.0,
            wmax=1.0,
            norm=100.0,
            reduction=torch.sum,
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
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * 250)]
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
    weights_im = None

    t0 = time()
    for step, batch in enumerate(tqdm(dataloader)):
        # Prep next input batch.
        inputs = batch["encoded_image"]

        inpts = {"I": inputs}
        if args.gpu:
            inpts = {k: v.cuda() for k, v in inpts.items()}

        clamp = torch.nn.functional.one_hot(batch["label"], num_classes=10).byte()
        unclamp = ~clamp
        clamp = {"O": clamp}
        unclamp = {"O": unclamp}

        # Run the network on the input.
        network.run(
            inpts=inpts,
            time=args.time,
            one_step=args.one_step,
            clamp=clamp,
            unclamp=unclamp,
        )

        if args.plot:
            # Plot output spikes.
            spikes = {l: network.monitors[l].get("s")[:, 0] for l in network.monitors}
            spike_ims, spike_axes = plot_spikes(
                spikes=spikes, ims=spike_ims, axes=spike_axes
            )

            # Plot connection weights.
            weights = network.connections["I", "O"].w
            weights = get_square_weights(weights, n_sqrt=4, side=28)
            weights_im = plot_weights(
                weights, wmax=network.connections["I", "O"].wmax, im=weights_im
            )

            plt.pause(1e-2)

        # Reset state variables.
        network.reset_()

    network.learning = False

    for step, batch in enumerate(tqdm(dataloader)):
        # Prep next input batch.
        inputs = batch["encoded_image"]

        inpts = {"I": inputs}
        if args.gpu:
            inpts = {k: v.cuda() for k, v in inpts.items()}

        # Run the network on the input.
        network.run(inpts=inpts, time=args.time, one_step=args.one_step)

        if args.plot:
            # Plot output spikes.
            spikes = {l: network.monitors[l].get("s")[:, 0] for l in network.monitors}
            spike_ims, spike_axes = plot_spikes(
                spikes=spikes, ims=spike_ims, axes=spike_axes
            )

            # Plot connection weights.
            weights = network.connections["I", "O"].w
            weights = get_square_weights(weights, n_sqrt=4, side=28)
            weights_im = plot_weights(
                weights, wmax=network.connections["I", "O"].wmax, im=weights_im
            )

            plt.pause(1e-2)

    t1 = time() - t0

    print(f"Time: {t1}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--one-step", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
