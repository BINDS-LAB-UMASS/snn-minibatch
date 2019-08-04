import os
import argparse

import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST

from minibatch import ROOT_DIR
from minibatch.conversion.mlp import MLP


def main(args):
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_workers == -1:
        args.n_workers = args.gpu * 4 * torch.cuda.device_count()

    device = torch.device("cuda" if args.gpu else "cpu")

    # Get the MNIST data.
    kwargs = {"num_workers": 1, "pin_memory": True} if args.gpu else {}
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

    # Load trained MLP from disk.
    ann = MLP().to(device)
    f = os.path.join(args.job_dir, "ann.pt")
    ann.load_state_dict(state_dict=torch.load(f=f))

    # Testing.
    ann.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # Forward pass.
            output = ann(data)

            # Get the index of the max log-probability.
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(
        "\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
