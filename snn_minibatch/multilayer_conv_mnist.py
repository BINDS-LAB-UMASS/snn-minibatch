"""
Attempt to replicate the network and unsupervised training procedure from "Training Deep
Spiking Convolutional Neural Networks With STDP-Based Unsupervised Pre-training Followed
by Supervised Fine-Tuning"
(https://www.frontiersin.org/articles/10.3389/fnins.2018.00435/full). They use a layer-
wise pre-training scheme, which is not yet implemented in this script.
"""

import os
import argparse

import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms

import bindsnet
from bindsnet.utils import im2col_indices
from bindsnet.encoding import PoissonEncoder
from bindsnet.network.monitors import Monitor
from bindsnet.datasets import DataLoader, MNIST
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Conv2dConnection, Connection
from bindsnet.analysis.plotting import plot_spikes, plot_conv2d_weights, plot_weights

matplotlib.use("Qt5Agg")


class WeightDependentPost(bindsnet.learning.LearningRule):
    def __init__(self, connection, nu=None, reduction=None, weight_decay=0.0, **kwargs):
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        assert self.source.traces, "Pre-synaptic nodes must record spike traces."
        assert (
            connection.wmin != -np.inf and connection.wmax != np.inf
        ), "Connection must define finite wmin and wmax."

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        if isinstance(connection, Connection):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.x_offset = kwargs.get("x_offset", 0.2)

    def _connection_update(self, **kwargs) -> None:
        batch_size = self.source.batch_size

        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()

        update = 0

        # Post-synaptic update.
        if self.nu[1]:
            post = self.reduction(torch.bmm(source_x, target_s), dim=0)
            update = (
                self.nu[1] * (post - self.x_offset) * (self.wmax - self.connection.w)
            )

        self.connection.w += update

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # Get convolutional layer parameters.
        (
            out_channels,
            in_channels,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        update = 0

        # Post-synaptic update.
        if self.nu[1]:
            post = self.reduction(
                torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
            ).view(self.connection.w.size())
            update += (
                self.nu[1]
                * (post - self.x_offset)
                * (self.connection.w - self.wmin)
                * (self.wmax - self.connection.wmin)
            )

        self.connection.w += update

        super().update()


class SpatialPooling2dConnection(bindsnet.network.topology.AbstractConnection):
    def __init__(
        self,
        source,
        target,
        kernel_size,
        stride,
        nu=None,
        reduction=None,
        weight_decay=0.0,
        **kwargs,
    ):
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        self.kernel_size = kernel_size
        self.stride = stride

    def compute(self, s):
        pooled = F.avg_pool2d(
            s.float(), kernel_size=self.kernel_size, stride=self.stride
        )
        return (pooled >= 0.8).float()

    def update(self, **kwargs):
        pass

    def reset_(self):
        pass

    def normalize(self):
        pass


class PassThroughNodes(bindsnet.network.nodes.Nodes):
    def __init__(
        self,
        n=None,
        shape=None,
        traces=False,
        traces_additive=False,
        tc_trace=5e-2,
        trace_scale=1.0,
        sum_input=False,
    ):
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

    def forward(self, x: torch.Tensor):
        self.s = x

    def reset_(self):
        self.s.zero_()


def main(args):
    # Random seed.
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)

    # Determines number of workers.
    if args.n_workers == -1:
        args.n_workers = args.gpu * 4 * torch.cuda.device_count()

    # Build network.
    network = bindsnet.network.Network(dt=args.dt, batch_size=args.batch_size)

    # Layers.
    input_layer = Input(shape=(1, 28, 28), traces=True)
    conv1_layer = LIFNodes(shape=(20, 24, 24), traces=True)
    pool1_layer = PassThroughNodes(shape=(20, 12, 12), traces=True)
    conv2_layer = LIFNodes(shape=(50, 8, 8), traces=True)
    pool2_layer = PassThroughNodes(shape=(50, 4, 4), traces=True)
    dense_layer = LIFNodes(shape=(200,), traces=True)
    output_layer = LIFNodes(shape=(10,), traces=True)

    network.add_layer(input_layer, name="I")
    network.add_layer(conv1_layer, name="C1")
    network.add_layer(pool1_layer, name="P1")
    network.add_layer(conv2_layer, name="C2")
    network.add_layer(pool2_layer, name="P2")
    network.add_layer(dense_layer, name="D")
    network.add_layer(output_layer, name="O")

    # Connections.
    conv1_connection = Conv2dConnection(
        source=input_layer,
        target=conv1_layer,
        update_rule=WeightDependentPost,
        nu=(0.0, args.nu),
        kernel_size=5,
        stride=1,
        wmin=-1.0,
        wmax=1.0,
    )
    pool1_connection = SpatialPooling2dConnection(
        source=conv1_layer, target=pool1_layer, kernel_size=2, stride=2
    )
    conv2_connection = Conv2dConnection(
        source=pool1_layer,
        target=conv2_layer,
        update_rule=WeightDependentPost,
        nu=(0.0, args.nu),
        kernel_size=5,
        stride=1,
        wmin=-1.0,
        wmax=1.0,
    )
    pool2_connection = SpatialPooling2dConnection(
        source=conv2_layer, target=pool2_layer, kernel_size=2, stride=2
    )
    dense_connection = Connection(
        source=pool2_layer,
        target=dense_layer,
        update_rule=WeightDependentPost,
        nu=(0.0, args.nu),
        wmin=-1.0,
        wmax=1.0,
    )
    output_connection = Connection(
        source=dense_layer,
        target=output_layer,
        update_rule=WeightDependentPost,
        nu=(0.0, args.nu),
        wmin=-1.0,
        wmax=1.0,
    )

    network.add_connection(connection=conv1_connection, source="I", target="C1")
    network.add_connection(connection=pool1_connection, source="C1", target="P1")
    network.add_connection(connection=conv2_connection, source="P1", target="C2")
    network.add_connection(connection=pool2_connection, source="C2", target="P2")
    network.add_connection(connection=dense_connection, source="P2", target="D")
    network.add_connection(connection=output_connection, source="D", target="O")

    # Monitors.
    for name, layer in network.layers.items():
        monitor = Monitor(obj=layer, state_vars=("s",), time=args.time)
        network.add_monitor(monitor=monitor, name=name)

    # Directs network to GPU.
    if args.gpu:
        network.to("cuda")

    # Load MNIST data.
    dataset = MNIST(
        PoissonEncoder(time=args.time, dt=args.dt),
        None,
        root=os.path.join(bindsnet.ROOT_DIR, "data", "MNIST"),
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
        ),
    )

    spike_ims = None
    spike_axes = None
    conv1_weights_im = None
    conv2_weights_im = None
    dense_weights_im = None
    output_weights_im = None

    for epoch in range(args.n_epochs):
        # Create a dataloader to iterate over dataset.
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=args.gpu,
        )

        for step, batch in enumerate(tqdm(dataloader)):
            # Prep next input batch.
            inpts = {"I": batch["encoded_image"]}
            if args.gpu:
                inpts = {k: v.cuda() for k, v in inpts.items()}

            # Run the network on the input.
            network.run(inpts=inpts, time=args.time)

            # Plot simulation data.
            if args.plot:
                spikes = {}
                for name, monitor in network.monitors.items():
                    spikes[name] = monitor.get("s")[:, 0].view(args.time, -1)

                spike_ims, spike_axes = plot_spikes(
                    spikes, ims=spike_ims, axes=spike_axes
                )

                conv1_weights_im = plot_conv2d_weights(
                    conv1_connection.w, im=conv1_weights_im, wmin=-1.0, wmax=1.0
                )
                conv2_weights_im = plot_conv2d_weights(
                    conv2_connection.w, im=conv2_weights_im, wmin=-1.0, wmax=1.0
                )
                dense_weights_im = plot_weights(
                    dense_connection.w, im=dense_weights_im, wmin=-1.0, wmax=1.0
                )
                output_weights_im = plot_weights(
                    output_connection.w, im=output_weights_im, wmin=-1.0, wmax=1.0
                )

                plt.pause(1e-8)

            # Reset state variables.
            network.reset_()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--time", type=int, default=25)
    parser.add_argument("--intensity", type=int, default=128)
    parser.add_argument("--nu", type=float, default=1e-4)
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
