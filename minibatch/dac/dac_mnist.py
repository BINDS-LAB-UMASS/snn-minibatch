import os
import shutil
import argparse
from time import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015v2
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights
from bindsnet.analysis.plotting import plot_spikes, plot_weights
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels

from minibatch.util import colorize, max_without_indices


def main(args):
    if args.update_steps is None:
        args.update_steps = max(250 // args.batch_size, 1)

    update_interval = args.update_steps * args.batch_size

    # Sets up GPU use
    torch.backends.cudnn.benchmark = False
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)

    # Determines number of workers to use
    if args.n_workers == -1:
        args.n_workers = args.gpu * 4 * torch.cuda.device_count()

    n_sqrt = int(np.ceil(np.sqrt(args.n_neurons)))

    if args.reduction == "sum":
        reduction = torch.sum
    elif args.reduction == "mean":
        reduction = torch.mean
    elif args.reduction == "max":
        reduction = max_without_indices
    else:
        raise NotImplementedError

    # Build network.
    network = DiehlAndCook2015v2(
        n_inpt=784,
        n_neurons=args.n_neurons,
        inh=args.inh,
        dt=args.dt,
        norm=78.4,
        nu=(1e-4, 1e-2),
        reduction=reduction,
        theta_plus=args.theta_plus,
        inpt_shape=(1, 28, 28),
    )

    # Directs network to GPU
    if args.gpu:
        network.to("cuda")

    # Load MNIST data.
    dataset = MNIST(
        PoissonEncoder(time=args.time, dt=args.dt),
        None,
        root=os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
        ),
    )

    test_dataset = MNIST(
        PoissonEncoder(time=args.time, dt=args.dt),
        None,
        root=os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * args.intensity)]
        ),
    )

    # Neuron assignments and spike proportions.
    n_classes = 10
    assignments = -torch.ones(args.n_neurons)
    proportions = torch.zeros(args.n_neurons, n_classes)
    rates = torch.zeros(args.n_neurons, n_classes)

    # Set up monitors for spikes and voltages
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=args.time)
        network.add_monitor(spikes[layer], name="%s_spikes" % layer)

    weights_im = None
    spike_ims, spike_axes = None, None

    # Record spikes for length of update interval.
    spike_record = torch.zeros(update_interval, args.time, args.n_neurons)

    if os.path.isdir(args.log_dir):
        shutil.rmtree(args.log_dir)

    # Summary writer.
    writer = SummaryWriter(log_dir=args.log_dir, flush_secs=60)

    for epoch in range(args.n_epochs):
        print(f"\nEpoch: {epoch}\n")

        labels = []

        # Create a dataloader to iterate and batch data
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            pin_memory=args.gpu,
        )

        for step, batch in enumerate(dataloader):
            print(f"Step: {step}")

            global_step = 60000 * epoch + args.batch_size * step

            if step % args.update_steps == 0 and step > 0:
                # Convert the array of labels into a tensor
                label_tensor = torch.tensor(labels)

                # Get network predictions.
                all_activity_pred = all_activity(
                    spikes=spike_record, assignments=assignments, n_labels=n_classes
                )
                proportion_pred = proportion_weighting(
                    spikes=spike_record,
                    assignments=assignments,
                    proportions=proportions,
                    n_labels=n_classes,
                )

                writer.add_scalar(
                    tag="accuracy/all vote",
                    scalar_value=torch.mean(
                        (label_tensor.long() == all_activity_pred).float()
                    ),
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="accuracy/proportion weighting",
                    scalar_value=torch.mean(
                        (label_tensor.long() == proportion_pred).float()
                    ),
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="spikes/mean",
                    scalar_value=torch.mean(torch.sum(spike_record, dim=1)),
                    global_step=global_step,
                )

                square_weights = get_square_weights(
                    network.connections["X", "Y"].w.view(784, args.n_neurons),
                    n_sqrt,
                    28,
                )
                img_tensor = colorize(square_weights, cmap="hot_r")

                writer.add_image(
                    tag="weights",
                    img_tensor=img_tensor,
                    global_step=global_step,
                    dataformats="HWC",
                )

                # Assign labels to excitatory layer neurons.
                assignments, proportions, rates = assign_labels(
                    spikes=spike_record,
                    labels=label_tensor,
                    n_labels=n_classes,
                    rates=rates,
                )

                labels = []

            labels.extend(batch["label"].tolist())

            # Prep next input batch.
            inpts = {"X": batch["encoded_image"]}
            if args.gpu:
                inpts = {k: v.cuda() for k, v in inpts.items()}

            # Run the network on the input.
            t0 = time()
            network.run(inpts=inpts, time=args.time, one_step=args.one_step)
            t1 = time() - t0

            # Add to spikes recording.
            s = spikes["Y"].get("s").permute((1, 0, 2))
            spike_record[
                (step * args.batch_size)
                % update_interval : (step * args.batch_size % update_interval)
                + s.size(0)
            ] = s

            writer.add_scalar(
                tag="time/simulation", scalar_value=t1, global_step=global_step
            )

            # Plot simulation data.
            if args.plot:
                input_exc_weights = network.connections["X", "Y"].w
                square_weights = get_square_weights(
                    input_exc_weights.view(784, args.n_neurons), n_sqrt, 28
                )
                spikes_ = {layer: spikes[layer].get("s")[:, 0] for layer in spikes}
                spike_ims, spike_axes = plot_spikes(
                    spikes_, ims=spike_ims, axes=spike_axes
                )
                weights_im = plot_weights(square_weights, im=weights_im)

                plt.pause(1e-8)

            # Reset state variables.
            network.reset_()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-neurons", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reduction", type=str, default="sum")
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--update-steps", type=int, default=None)
    parser.add_argument("--inh", type=float, default=120)
    parser.add_argument("--theta_plus", type=float, default=0.05)
    parser.add_argument("--time", type=int, default=100)
    parser.add_argument("--dt", type=int, default=1.0)
    parser.add_argument("--intensity", type=float, default=128)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--one-step", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
