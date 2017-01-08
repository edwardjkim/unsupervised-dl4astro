#!/usr/bin/env python3
import argparse
from pixelsg.train import train_cnn


def main(args=None):

    network = train_cnn(
        filenames=args.files,
        num_epochs=args.num_epochs,
        num_classes=args.num_classes,
        size=args.size
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Trains an unsupervised convolutional neural network on "
            "FITS images."
    )

    parser.add_argument(
        "files", metavar="FILES", type=str, nargs='+',
        help="file names"
    )

    parser.add_argument(
        "--num_epochs", type=int, default=500,
        help="Number of training epochs to perform (default: 500)"
    )

    parser.add_argument(
        "--num_classes", type=int, default=1000,
        help="Number of surrogate classes (default: 1000)"
    )

    parser.add_argument(
        "--size", type=int, default=32,
        help="Size of a patch in number of pixels (default: 32)"
    )

    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="File name of a pre-trained model (default: None)"
    )

    args = parser.parse_args()

    main(args)
