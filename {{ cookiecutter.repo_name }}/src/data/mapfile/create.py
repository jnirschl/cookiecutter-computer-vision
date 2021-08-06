#!/usr/bin/env python3

import os
import logging
import pandas as pd
from pathlib import Path
import yaml

from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator

# set tf warning options
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def create(
    input_dir,
    output_dir,
    output_filename="mapfile_df.csv",
    na_rep="nan",
    target_size=(224, 224, 3),
    color_mode="rgb",
    save_format="png",
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
