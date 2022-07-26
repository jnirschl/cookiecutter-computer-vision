import logging
from pathlib import Path

import click
import cv2
import numpy as np
from dotenv import find_dotenv, load_dotenv
from skimage.filters import threshold_multiotsu

from src.img.morphology import bwareaopen
from src.img.segmentation import filters, watershed

cv2.setUseOptimized(True)


def threshold_single(
    image,
    blur=None,
    blur_kernel=5,
    thresh=0,
    mode=cv2.THRESH_OTSU,
    adaptive=None,
    watershed_flag=True,
    min_distance=20,
    min_size=20,
    fill_holes=True,
    morphology_flag=True,
    # output_dir: str,
    # params_filepath: str = "params.yaml",
    # grayscale: bool = True,
):
    """sdfs"""

    if blur:
        image = filters.smooth(image, blur=blur.lower(), kernel=blur_kernel)

    # threshold
    if adaptive is None:
        ret, mask = cv2.threshold(image, thresh, 255, mode)
    elif adaptive.lower() == "mean":
        mask = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif adaptive.lower() == "gauss":
        mask = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif adaptive.lower() == "multiotsu":
        mask = threshold_multiotsu(image=thresh, classes=2, nbins=256, hist=None)
    else:
        ret, mask = cv2.threshold(image, thresh, 255, mode)

    # morphology
    if morphology_flag:
        erosion = cv2.erode(image, np.ones((1, 1), np.uint8), iterations=1)
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    # bwareaopen
    if min_size > 0:
        mask = bwareaopen(mask, min_size, connectivity=8)

    # watershed
    if watershed_flag:
        mask = watershed(mask, min_distance=min_distance)
        mask = mask > 0

    # fill holes
    if fill_holes:
        mask_tmp = np.zeros(tuple(np.add(mask.shape, 2)), np.uint8)
        cv2.floodFill(mask.astype(np.uint8), mask_tmp, (0, 0), 0)
        mask = cv2.bitwise_not(mask_tmp[1:-1, 1:-1]).astype(np.uint8)
        mask = cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX)

    return mask


@click.command()
@click.argument(
    "input_dir",
    type=click.File(exists=True),
)
@click.argument(
    "output_dir",
    default=Path("./data/processed").resolve(),
    type=click.Path(exists=True),
)
@click.argument("output_filename", default="mapfile_df.csv", type=click.Path())
@click.option("--params_filepath", "-p", default="params.yaml")
@click.option(
    "--force",
    is_flag=True,
    help="Switch to force overwrite existing mean image.",
)
@click.option("--na-rep", default="nan")
def main(
    input_dir,
    output_dir,
    params_filepath="params.yaml",
    force=True,
):
    threshold_single(
        input_dir, output_dir, params_filepath=params_filepath, force=force
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
