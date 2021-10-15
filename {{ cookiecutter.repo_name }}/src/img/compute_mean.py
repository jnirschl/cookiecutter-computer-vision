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
    # assert img_shape is None or (img_shape is tuple and len(img_shape) == 3), TypeError(
    #     f"IMG_SHAPE must be tuple with a length of 3"
    # )
    # setup logging
    logger = logging.getLogger(__name__)

    # set image flag
    FORMAT = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    # check if file exists
    if not force and Path(mapfile).parent.joinpath("mean_image.png").exists():
        mean_img_path = Path(mapfile).parent.joinpath("mean_image.png")
        logger.info(f"Using existing mean image:\n{mean_img_path}")
        return cv2.imread(mean_img_path, FORMAT)

    # read mapfile
    mapfile_df = pd.read_csv(mapfile, sep=",", header=0, index_col=0)

    if img_shape is None:
        # use shape of first image if none given
        img_shape = cv2.imread(mapfile_df["filename"][0], FORMAT).shape

    # compute mean and std
    mean_img = mean_subfun(mapfile_df, img_shape, FORMAT)
    std_img = std_subfun(mean_img, mapfile_df, img_shape, FORMAT)

    # save images
    mean_filename = str(Path(mapfile).parent.joinpath("mean_image.png"))
    cv2.imwrite(mean_filename, mean_img)
    std_filename = str(Path(mapfile).parent.joinpath("std_image.png"))
    cv2.imwrite(std_filename, std_img)
    return mean_img, std_img


def mean_subfun(mapfile_df, img_shape, FORMAT):
    """"""

    logger = logging.getLogger(__name__)
    logging_flag = False

    mean_img = np.zeros(img_shape, dtype=np.float32)

    # process files
    logger.info("Computing image mean")
    for idx, (filename, label) in mapfile_df.iterrows():
        # print(f"{idx}\t{filename}\t{label}")
        img = cv2.imread(filename, FORMAT)

        if img is None:
            logging.warning(
                f"Error opening file:\t{Path('./').joinpath(*Path(filename).parts[-3:])}"
            )

        if img.shape != mean_img.shape:
            img = np.reshape(img, mean_img.shape)

        if img.shape[0:2] != img_shape[0:2]:
            if not logging_flag:
                logger.info(f"Resizing images to: {img_shape}")  # print once
                logging_flag = True
            img = cv2.resize(img, img_shape[0:2], interpolation=cv2.INTER_AREA)

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
    return mean_img


def std_subfun(mean_img, mapfile_df, img_shape, FORMAT):
    """"""

    logger = logging.getLogger(__name__)
    logging_flag = False

    # normalize mean image
    mean_img = np.divide(mean_img, 255.0).astype(dtype=np.float64)
    std_img = np.zeros(img_shape, dtype=np.float64)

    # process files
    logger.info("Computing image standard deviation")
    for idx, (filename, label) in mapfile_df.iterrows():
        # print(f"{idx}\t{filename}\t{label}")
        img = cv2.imread(filename, FORMAT)

        if img is None:
            logging.warning(
                f"Error opening file:\t{Path('./').joinpath(*Path(filename).parts[-3:])}"
            )

        if img.shape != mean_img.shape:
            img = np.reshape(img, mean_img.shape)

        if tuple(img.shape[0:2]) != tuple(img_shape[0:2]):
            if not logging_flag:
                logger.info(f"Resizing images to: {img_shape}")  # print once
                logging_flag = True
            img = cv2.resize(img, img_shape[0:2], interpolation=cv2.INTER_AREA)

        # ensure image is valid
        if img is None:
            raise ValueError(f"Error loading image:\t{filename}")
        if idx % 1000 == 0 and idx > 0:
            print(f"Processed {idx} images.")

        # subtract img from mean_img and square the difference
        img_norm = np.divide(img, 255.0).astype(dtype=np.float64)
        diff_img_norm = np.subtract(img_norm, mean_img)
        sq_diff_img_norm = np.power(diff_img_norm, 2)

        # sum of the squared difference
        std_img = cv2.accumulate(sq_diff_img_norm.astype(dtype=np.float64), std_img)

    logger = logging.getLogger(__name__)
    logger.info(f"Processed {idx+1} images.")
    print(f"Processed {idx+1} images.")

    # divide by n_images
    std_img_float = np.sqrt(np.divide(std_img, idx + 1))
    std_img_uint8 = (std_img_float*255.0).astype(np.uint8)
    return std_img_uint8


@click.command()
@click.argument(
    "mapfile",
    default=Path("./data/processed/mapfile_df.csv").resolve(),
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
