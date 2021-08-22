#!/usr/bin/env python3

import os
from glob import glob
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
    segmentation=False,
    save_format=".png",
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

    if segmentation:
        filepaths, labels = segmentation_dir(input_dir, save_format=save_format)
    else:
        filepaths, labels = classification_dir(input_dir, output_dir)

    # create mapfile_df with a list of 0: relative filepath and 1:class labels
    if filepaths:
        mapfile_df = pd.DataFrame({"filename": filepaths, "class": labels})

        # save output mapfile_df
        mapfile_df.to_csv(
            output_dir.joinpath(output_filename),
            sep=",",
            na_rep=na_rep,
        )

        return mapfile_df
    else:
        # raise error if dir_iterator is empty
        return None
        # raise ValueError(f"No subdirectories with images identified in:\t{input_dir}")


def classification_dir(input_dir, output_dir):
    # initialize dummy instance of ImageDataGenerator
    datagen = ImageDataGenerator()

    # create instance of DirectoryIterator
    dir_iterator = DirectoryIterator(
        input_dir,
        datagen,
    )

    # save label encoding dictionary
    encoding_dict = yaml.safe_dump(dir_iterator.class_indices)
    with open(os.path.join(output_dir, "label_encoding.yaml"), "w") as writer:
        writer.writelines(encoding_dict)

    return dir_iterator.filepaths, dir_iterator.labels


def segmentation_dir(input_dir, save_format=".png"):

    # set vars
    image_prefix = "data"
    mask_prefix = "mask"
    input_dir = Path(input_dir)
    for subdir in [image_prefix, mask_prefix]:
        assert input_dir.joinpath(subdir).exists(), NotADirectoryError(
            f"Expected subdirectory name: {subdir}"
        )

    # glob image filepaths in folder 'data'
    image_dir = input_dir.joinpath(image_prefix, "*" + save_format)
    image_filepaths = sorted(glob(str(image_dir)))

    # glob image filepaths in folder 'data'
    mask_dir = input_dir.joinpath(mask_prefix, "*" + save_format)
    mask_filepaths = sorted(glob(str(mask_dir)))

    return image_filepaths, mask_filepaths
