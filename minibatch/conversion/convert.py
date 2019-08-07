import argparse
import os
from time import time

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from bindsnet.conversion import ann_to_snn
from bindsnet.encoding import RepeatEncoder
from bindsnet.datasets import MNIST, DataLoader

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

    # Load trained MLP from disk.
    ann = MLP().to(device)
    f = os.path.join(args.job_dir, "ann.pt")
    ann.load_state_dict(state_dict=torch.load(f=f))

    # Load dataset.
    dataset = MNIST(
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

    # Do ANN to SNN conversion.
    data = dataset.data.float()
    data /= data.max()
    data = data.view(-1, 784)
    snn = ann_to_snn(ann, input_shape=(784,), data=data.to(device))
    snn = snn.to(device)

    print(snn)

    # Create a dataloader to iterate and batch data
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=args.gpu,
    )

    correct = 0
    t0 = time()
    for step, batch in enumerate(tqdm(dataloader)):
        # Prep next input batch.
        inputs = batch["encoded_image"]
        labels = batch["label"]

        inpts = {"Input": inputs}
        if args.gpu:
            inpts = {k: v.cuda() for k, v in inpts.items()}

        # Run the network on the input.
        snn.run(inpts=inpts, time=args.time, one_step=args.one_step)

        output_voltages = snn.layers["5"].summed
        prediction = torch.softmax(output_voltages, dim=1).argmax(dim=1)
        correct += (prediction.cpu() == labels).sum().item()

        # Reset state variables.
        snn.reset_()

    t1 = time() - t0

    accuracy = 100 * correct / len(dataloader.dataset)

    print(f"SNN accuracy: {accuracy:.2f}")

    path = os.path.join(ROOT_DIR, "results", args.results_file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.isfile(path):
        with open(os.path.join(path), "w") as f:
            f.write("seed,simulation time,batch size,inference time,accuracy\n")

    to_write = [args.seed, args.time, args.batch_size, t1, accuracy]
    to_write = ",".join(map(str, to_write)) + "\n"
    with open(os.path.join(path), "a") as f:
        f.write(to_write)

    return t1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--results-file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--n-workers", type=int, default=-1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--one-step", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
