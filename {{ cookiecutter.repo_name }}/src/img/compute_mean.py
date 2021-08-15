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


def image(mapfile, img_shape=None, grayscale=False, force=True):
    """Accept str with full filepath to the mapfile, compute mean over all images,
    and write output mean image as png file."""
    assert type(mapfile) is str, TypeError(f"MAPFILE must be type STR")
    assert img_shape is None or (img_shape is tuple and len(img_shape) == 3), TypeError(
        f"IMG_SHAPE must be tuple with a length of 3"
    )

    # set image flag
    FORMAT = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    # check if file exists
    if not force and Path(mapfile).parent.joinpath("mean_image.png").exists():
        mean_img_path = Path(mapfile).parent.joinpath("mean_image.png")
        logger = logging.getLogger(__name__)
        logger.info(f"Using existing mean image:\n\t{mean_img_path}")
        return cv2.imread(mean_img_path, FORMAT)

    # read mapfile
    mapfile_df = pd.read_csv(mapfile, sep=",", header=0, index_col=0)

    # pre-allocate mean image
    if img_shape is None:
        # use shape of first image if none given
        img_shape = cv2.imread(mapfile_df["filename"][0], FORMAT).shape

    mean_img = np.zeros(img_shape, dtype=np.float32)

    logger = logging.getLogger(__name__)
    logger.info("Computing mean image")

    # process files
    for idx, (filename, label) in mapfile_df.iterrows():
        # print(f"{idx}\t{filename}\t{label}")
        img = cv2.imread(filename, FORMAT)

        # ensure image is valid
        if img is None:
            raise ValueError(f"Error loading image:\t{filename}")
        if idx % 1000 == 0 and idx > 0:
            print(f"Processed {idx} images.")

        # accumulate
        mean_img = cv2.accumulate(img.astype(dtype=np.float32), mean_img)

    logger = logging.getLogger(__name__)
    logger.info(f"Processed {idx+1} images.")
    print(f"Processed {idx+1} images.")

    # divide by n_images
    mean_img = np.divide(mean_img, idx + 1).astype(np.uint8)

    # save image
    output_filename = str(Path(mapfile).parent.joinpath("mean_image.png"))
    cv2.imwrite(output_filename, mean_img)
    return mean_img


@click.command()
@click.argument(
    "mapfile",
    default=Path("./data/interim/mapfile.csv").resolve(),
    type=click.Path(exists=True),
)
@click.option("--img-shape", default=None, help="Tuple with image dimensions.")
@click.option(
    "--grayscale",
    is_flag=True,
    help="Flag to read images as 8-bit grayscale.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Switch to force overwrite existing images.",
)
def main(mapfile, grayscale, force, img_shape=None):
    """Click interface"""
    image(mapfile, img_shape=img_shape, grayscale=grayscale, force=force)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(mapfile, img_shape=None, grayscale=False)
