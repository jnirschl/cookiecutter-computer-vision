#!/usr/bin/env python3

import logging
import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

# load custom libraries from src
from src.data import mapfile
from src.img import compute_mean

# set tf warning options
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@click.command()
@click.option(
    "--force", default=False, help="Force overwrite of mapfile_df and mean_img.png."
)
@click.argument(
    "input_dir",
    default=Path("./data/raw").resolve(),
    type=click.Path(exists=True),
)
@click.argument(
    "output_dir",
    default=Path("./data/interim").resolve(),
    type=click.Path(),
)
@click.argument("output_filename", default="mapfile_df.csv", type=click.Path())
def main(
    input_dir,
    output_dir,
    output_filename="mapfile_df.csv",
    params_filepath="params.yaml",
    force=False,
    na_rep="nan",
):

    """Runs data processing scripts to accept a directory INPUT_DIR containing
    sub-folders of images with one sub-folder per class and create a CSV file
    mapping the image filepath to integer class labels, which is saved in
    OUTPUT_DIR.
    """

    # create mapfile_df
    mapfile_df = mapfile.create(input_dir, output_dir, output_filename, na_rep)

    # compute mean image
    compute_mean.image(
        mapfile_df,
        output_dir=output_dir,
        force=force,
        params_filepath=params_filepath,
        color_mode="gray",
    )

    # split train test dev
    mapfile.split(mapfile_df, "./data/processed")
    return mapfile_df


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
