import os
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
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
    ann = MLP().to(device)

    # Specify optimizer and loss function.
    optimizer = optim.Adam(params=ann.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # Train / test the ANN.
    best_accuracy = -np.inf
    for epoch in range(1, args.n_epochs + 1):

        # Training.
        ann.train()
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = ann(data)
            loss = criterion(output, target)

            # Get the index of the max log-probability.
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

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

        print(
            "\nTrain accuracy: {:.2f}%".format(
                100.0 * correct / len(train_loader.dataset)
            )
        )

        # Testing.
        ann.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                # Forward pass.
                output = ann(data)

                # Sum batch loss.
                test_loss += criterion(output, target).item()

                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        accuracy = 100.0 * correct / len(test_loader.dataset)
        if accuracy > best_accuracy:
            # Save model to disk.
            f = os.path.join(args.job_dir, "ann.pt")
            os.makedirs(os.path.dirname(f), exist_ok=True)
            torch.save(ann.state_dict(), f=f)

        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), accuracy
            )
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
