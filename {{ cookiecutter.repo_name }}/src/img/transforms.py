#!/usr/bin/env python3

import os
import click
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Specify opencv optimization
from dotenv import find_dotenv, load_dotenv

cv2.setUseOptimized(True)


@click.command()
@click.option(
    "--grayscale", default=True, help="Switch to read images as 8-bit grayscale."
)
@click.argument(
    "mapfile",
    default=Path("./data/interim/mapfile.csv").resolve(),
    type=click.Path(exists=True),
)
# @click.argument(
#     "output_dir",
#     default=Path("./data/interim").resolve(),
#     type=click.Path(),
# )
def mean_image(mapfile, img_shape=None, grayscale=False):
    """Accept images as numpy array with images separated by rows
    and columns indicating pixel values"""
    assert type(mapfile) is str, TypeError(f"MAPFILE must be type STR")
    assert img_shape is None or (img_shape is tuple and len(img_shape) == 3), TypeError(
        f"IMG_SHAPE must be tuple with a length of 3"
    )

    # set image flag
    if grayscale:
        FORMAT = cv2.IMREAD_GRAYSCALE
    else:
        FORMAT = cv2.IMREAD_COLOR

    # read mapfile
    mapfile_df = pd.read_csv(mapfile, sep=",", header=0, index_col=0)

    # pre-allocate mean image
    if img_shape is None:
        # use shape of first image if none given
        img_shape = cv2.imread(mapfile_df["filename"][0], FORMAT).shape

    mean_img = np.zeros(img_shape, dtype=np.float32)

    # process files
    print(f"Computing mean image...")
    for idx, (filename, label) in mapfile_df.iterrows():
        # print(f"{idx}\t{filename}\t{label}")
        img = cv2.imread(filename, FORMAT)

        # ensure image is valid
        if img is None:
            raise ValueError(f"Error loading image:\t{filename}")
        else:
            if idx % 1000 == 0:
                print(f"Processed {idx} images.")

            # accumulate
            mean_img = cv2.accumulate(img.astype(dtype=np.float32), mean_img)

    # divide by n_images
    print(f"Processed {idx+1} images.")
    mean_img = np.divide(mean_img, idx + 1)

    # save image
    output_filename = str(Path(mapfile).parent.joinpath("mean_image.png"))
    cv2.imwrite(output_filename, mean_img)
    return


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    mean_image()
