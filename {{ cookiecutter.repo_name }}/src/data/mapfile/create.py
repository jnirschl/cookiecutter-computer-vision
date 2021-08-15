#!/usr/bin/env python3

import os
import logging
import pandas as pd
from pathlib import Path
import yaml

from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

from src.data import load_params

# set tf warning options
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create(
    input_dir,
    output_dir,
    output_filename="mapfile_df.csv",
    na_rep="nan",
    target_size=None,
    color_mode=None,
    save_format=None,
    label_dict_name="label_encoding.yaml",
):

    """Runs data processing scripts to accept a directory INPUT_DIR containing
    sub-folders of images with one sub-folder per class and create a CSV file
    mapping the image filepath to integer class labels, which is saved in
    OUTPUT_DIR.
    """
    # resolve relative paths and ensure directors are uniform of type Path()
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()

    logger = logging.getLogger(__name__)
    logger.info("Creating mapfile_df from directory structure")

    # initialize dummy instance of ImageDataGenerator
    datagen = ImageDataGenerator()

    # load params and set optional input args using params.yaml, if none given
    params = load_params()
    target_size, save_format, color_mode = set_options(
        params, target_size, save_format, color_mode
    )

    # create instance of DirectoryIterator
    dir_iterator = DirectoryIterator(
        input_dir,
        datagen,
        target_size=target_size,
        color_mode=color_mode,
        save_format=save_format,
    )

    # create mapfile_df with a list of 0: relative filepath and 1:class labels
    if dir_iterator.filepaths:
        mapfile_df = pd.DataFrame(
            {"filename": dir_iterator.filepaths, "class": dir_iterator.labels}
        )

        # save output mapfile_df
        mapfile_df.to_csv(
            output_dir.joinpath(output_filename),
            sep=",",
            na_rep=na_rep,
        )

        # save label encoding dictionary
        encoding_dict = yaml.safe_dump(dir_iterator.class_indices)
        with open(os.path.join(output_dir, label_dict_name), "w") as writer:
            writer.writelines(encoding_dict)

        return mapfile_df
    else:
        # raise error if dir_iterator is empty
        return None
        # raise ValueError(f"No subdirectories with images identified in:\t{input_dir}")


def set_options(params, target_size, save_format, color_mode):
    """Sub-function to set optional input arguments"""

    if target_size is None:
        target_size = tuple(params["flow_from_dataframe"]["target_size"])

    if save_format is None:
        save_format = params["flow_from_dataframe"]["save_format"]

    if color_mode is None:
        color_mode = params["flow_from_dataframe"]["color_mode"]

    # validate argument properties
    assert type(target_size) is tuple, TypeError(f"TARGET_SIZE must be type{tuple}")
    assert save_format in ["jpg", "png"], TypeError(f"SAVE_FORMAT must be 'jpg or png'")
    assert color_mode in ["rgb", "rgba", "grayscale"], TypeError(
        "COLOR_MODE must be 'rgb, rgba, or grayscale'"
    )

    return target_size, save_format, color_mode
