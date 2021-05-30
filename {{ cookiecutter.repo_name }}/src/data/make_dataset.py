#!/usr/bin/env python3

import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd

import tensorflow as tf

# set tf warning options
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@click.command()
# @click.option(
#     "--force", default=False, help="Overwrite of mapfile if it already exists."
# )
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
@click.argument("output_filename", default="mapfile.csv", type=click.Path())
def create_mapfile(
    input_dir, output_dir, output_filename="mapfile.csv", na_rep="nan"
):

    """Runs data processing scripts to accept a directory INPUT_DIR containing
    sub-folders of images with one sub-folder per class and create a CSV file
    mapping the image filepath to integer class labels, which is saved in
    OUTPUT_DIR.
    """
    logger = logging.getLogger(__name__)
    logger.info("making interim data set from raw data")

    # resolve relative paths
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # initialize dummy instance of ImageDataGenerator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    # create instance of DirectoryIterator
    dir_iterator = tf.keras.preprocessing.image.DirectoryIterator(input_dir, datagen)

    # create mapfile with a list of 0: relative filepath and 1:class labels
    if dir_iterator.filepaths:
        mapfile_df = pd.DataFrame(
            {"filename": dir_iterator.filepaths, "label": dir_iterator.labels}
        )

        # save output mapfile
        mapfile_df.to_csv(output_dir.joinpath(output_filename), na_rep=na_rep)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    create_mapfile()
