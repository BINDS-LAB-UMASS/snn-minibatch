import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from time import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from bindsnet.datasets import MNIST as BNMNIST, DataLoader
from bindsnet.conversion import ann_to_snn
from bindsnet.encoding import RepeatEncoder
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes

from . import ROOT_DIR


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def main(args):
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_workers == -1:
        args.n_workers = args.gpu * 4 * torch.cuda.device_count()

    print()
    print("Creating and training the ANN...")
    print()

    # Get the MNIST data.
    kwargs = {"num_workers": 1, "pin_memory": True} if args.gpu else {}
    train_dataset = MNIST(
        os.path.join(ROOT_DIR, "data"),
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )
    test_dataset = MNIST(
        os.path.join(ROOT_DIR, "data"),
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        ),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
    )

    # Create and train an ANN on the MNIST dataset.
    device = torch.device("cuda" if args.gpu else "cpu")
    ann = MLP().to(device)

    # Specify optimizer and loss function.
    optimizer = optim.Adam(params=ann.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Train / test the ANN.
    for epoch in range(1, args.n_epochs + 1):

        # Training...
        ann.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = ann(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

        # Testing...
        ann.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = ann(data)
                test_loss += criterion(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    print()
    print("Converting ANN to SNN...")
    print()

    # Do ANN to SNN conversion.
    data = train_dataset.data.float()
    data /= data.max()
    data = data.view(-1, 784)
    snn = ann_to_snn(ann, input_shape=(784,), data=data.to(device))
    snn = snn.to(device)

    print(snn)

    for layer in snn.layers:
        if layer != "Input":
            snn.add_monitor(
                Monitor(snn.layers[layer], state_vars=["s", "v"], time=args.time),
                name=layer,
            )

    spike_ims = None
    spike_axes = None

    dataset = BNMNIST(
        image_encoder=RepeatEncoder(time=args.time, dt=1.0),
        label_encoder=None,
        root=os.path.join(ROOT_DIR, "data", "MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        ),
    )

    # Create a dataloader to iterate and batch data
    dataloader = DataLoader(
        dataset,
        batch_size=args.snn_batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=args.gpu,
    )

    print()
    print("Testing SNN on MNIST data...")
    print()

    means = []
    for step, batch in enumerate(tqdm(dataloader)):
        # Prep next input batch.
        inputs = batch["encoded_image"]
        labels = batch["label"]

        inpts = {"Input": inputs}
        if args.gpu:
            inpts = {k: v.cuda() for k, v in inpts.items()}

        # Run the network on the input.
        t0 = time()
        snn.run(inpts=inpts, time=args.time)
        t1 = time() - t0

        spikes = {layer: monitor.get("s") for layer, monitor in snn.monitors.items()}
        voltages = {layer: monitor.get("v") for layer, monitor in snn.monitors.items()}
        output_voltages = voltages["5"]
        prediction = torch.softmax(output_voltages.sum(dim=0), dim=1).argmax(dim=1)
        mean = (prediction.cpu() == labels).float().mean().item()
        means.append(mean)

        # Reset state variables.
        snn.reset_()

        if args.plot:
            spikes = {k: spikes[k].cpu() for k in spikes}
            spike_ims, spike_axes = plot_spikes(spikes, ims=spike_ims, axes=spike_axes)
            plt.pause(1e-3)

    print(f"SNN accuracy: {100 * np.mean(means):.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--snn-batch-size", type=int, default=100)
    parser.add_argument("--time", type=int, default=50)
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
