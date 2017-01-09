#!/usr/bin/env python3
import argparse
from pixelsg.train import train_cnn


def main(args=None):

    network = train_cnn(
        filenames=args.files,
        num_epochs=args.num_epochs,
        num_classes=args.num_classes,
        size=args.size,
        bands=args.bands
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
        "--num_epochs", type=int, default=1000,
        help="Number of training epochs to perform (default: 1000)"
    )

    parser.add_argument(
        "--num_classes", type=int, default=1000,
        help="Number of surrogate classes (default: 1000)"
    )

    parser.add_argument(
        "--size", type=int, default=16,
        help="Size of a patch in number of pixels (default: 16)"
    )

    parser.add_argument(
        "--bands", type=str, default="gri",
        help="ugriz or gri or r or grizy etc. (default: gri)"
    )

    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="File name of a pre-trained model (default: None)"
    )

    args = parser.parse_args()

    main(args)
