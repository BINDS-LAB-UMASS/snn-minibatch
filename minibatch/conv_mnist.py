import os
import argparse
from time import time

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.network import Network
from bindsnet.learning import PostPre
from bindsnet.encoding import PoissonEncoder
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_conv2d_weights
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Conv2dConnection, Connection

from .util import max_without_indices


def main(args):
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)

    conv_size = int((28 - args.kernel_size + 2 * args.padding) / args.stride) + 1

    # Build network.
    network = Network()
    input_layer = Input(n=784, shape=(1, 28, 28), traces=True)

    conv_layer = DiehlAndCookNodes(
        n=args.n_filters * conv_size * conv_size,
        shape=(args.n_filters, conv_size, conv_size),
        traces=True,
    )

    conv_conn = Conv2dConnection(
        input_layer,
        conv_layer,
        kernel_size=args.kernel_size,
        stride=args.stride,
        update_rule=PostPre,
        norm=0.4 * args.kernel_size ** 2,
        nu=[0, args.lr],
        reduction=max_without_indices,
        wmax=1.0,
    )

    w = torch.zeros(
        args.n_filters, conv_size, conv_size, args.n_filters, conv_size, conv_size
    )
    for fltr1 in range(args.n_filters):
        for fltr2 in range(args.n_filters):
            if fltr1 != fltr2:
                for i in range(conv_size):
                    for j in range(conv_size):
                        w[fltr1, i, j, fltr2, i, j] = -100.0

    w = w.view(
        args.n_filters * conv_size * conv_size, args.n_filters * conv_size * conv_size
    )
    recurrent_conn = Connection(conv_layer, conv_layer, w=w)

    network.add_layer(input_layer, name="X")
    network.add_layer(conv_layer, name="Y")
    network.add_connection(conv_conn, source="X", target="Y")
    network.add_connection(recurrent_conn, source="Y", target="Y")

    # Voltage recording for excitatory and inhibitory layers.
    voltage_monitor = Monitor(network.layers["Y"], ["v"], time=args.time)
    network.add_monitor(voltage_monitor, name="output_voltage")

    if args.gpu:
        network.to("cuda")

    # Load MNIST data.
    train_dataset = MNIST(
        PoissonEncoder(time=args.time, dt=args.dt),
        None,
        os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
        ),
    )

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=args.time)
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    voltages = {}
    for layer in set(network.layers) - {"X"}:
        voltages[layer] = Monitor(
            network.layers[layer], state_vars=["v"], time=args.time
        )
        network.add_monitor(voltages[layer], name="%s_voltages" % layer)

    # Train the network.
    print("Begin training.\n")
    start = time()

    weights_im = None

    for epoch in range(args.n_epochs):
        if epoch % args.progress_interval == 0:
            print(
                "Progress: %d / %d (%.4f seconds)"
                % (epoch, args.n_epochs, time() - start)
            )
            start = time()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=args.gpu,
        )

        for step, batch in enumerate(tqdm(train_dataloader)):
            # Get next input sample.
            inpts = {"X": batch["encoded_image"]}
            if args.gpu:
                inpts = {k: v.cuda() for k, v in inpts.items()}

            # Run the network on the input.
            network.run(inpts=inpts, time=args.time, input_time_dim=0)

            # Decay learning rate.
            network.connections["X", "Y"].nu[1] *= 0.99

            # Optionally plot various simulation information.
            if args.plot:
                weights = conv_conn.w
                weights_im = plot_conv2d_weights(weights, im=weights_im)

                plt.pause(1e-8)

            network.reset_()  # Reset state variables.

    print(
        "Progress: %d / %d (%.4f seconds)\n"
        % (args.n_epochs, args.n_epochs, time() - start)
    )
    print("Training complete.\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--n-filters", type=int, default=25)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--time", type=int, default=100)
    parser.add_argument("--dt", type=int, default=1.0)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--intensity", type=float, default=128.0)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--test", dest="train", action="store_false")
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--gpu", dest="gpu", action="store_true")
    parser.set_defaults(plot=False, gpu=False, train=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
