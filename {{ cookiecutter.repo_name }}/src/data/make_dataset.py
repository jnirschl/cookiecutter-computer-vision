#!/usr/bin/env python3

import logging
import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

# load custom libraries from src
from src.data import mapfile, load_params
from src.img import compute_mean

# set tf warning options
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create(
    input_dir,
    output_dir,
    output_filename="mapfile_df.csv",
    params_filepath="params.yaml",
    force=True,
    na_rep="nan",
):
    """Runs data processing scripts to accept a directory INPUT_DIR containing
    sub-folders of images with one sub-folder per class and create a CSV file
    mapping the image filepath to integer class labels, which is saved in
    OUTPUT_DIR.
    """

    # create mapfile_df
    mapfile_df = mapfile.create(input_dir, output_dir, output_filename, na_rep)
    mapfile_path = str(Path(output_dir).joinpath(output_filename).resolve())

    # load params_filepath
    params = load_params(params_filepath)
    img_shape = tuple(params["target_size"])
    grayscale = params["color_mode"].lower() == "grayscale"

    # compute mean image
    compute_mean.image(
        mapfile_path,
        img_shape=img_shape,  # TODO
        grayscale=grayscale,
        force=force,
        # params_filepath=params_filepath,
    )

    # split train test dev
    mapfile.split(mapfile_df, output_dir)
    return mapfile_df


@click.command()
@click.argument(
    "input_dir",
    default=Path("./data/raw").resolve(),
    type=click.Path(exists=True),
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
    output_filename="mapfile_df.csv",
    params_filepath="params.yaml",
    force=True,
    na_rep="nan",
):
    create(
        input_dir,
        output_dir,
        output_filename=output_filename,
        params_filepath=params_filepath,
        force=force,
        na_rep=na_rep,
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
