import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import load_data, load_params


def split(
    mapfile_df,
    output_dir=None,
    output_filename="split_train_dev.csv",
    params_filepath="params.yaml",
):
    """Accept a Pandas DataFrame with image filenames and labels,
    split into train, test, and dev sets and save file.

    Modify the n_split, the random_seed, or shuffle in params.yaml"""

    assert type(mapfile_df) is pd.DataFrame, TypeError(
        f"MAPFILE_DF must be type {pd.DataFrame}"
    )

    logger = logging.getLogger(__name__)
    logger.info("Creating train, dev, and test splits from mapfile_df")

    # set index
    mapfile_df.index.name = "index"

    # load params
    params = load_params(filepath=params_filepath)

    params_split = params["train_test_split"]
    params_split["random_seed"] = params["random_seed"]

    # get filenames and dependent variables (class)
    train_files = mapfile_df["filename"]
    if params["task"] == "segmentation":
        train_class = np.ones(len(mapfile_df["class"]), dtype=np.int32)
    elif params["task"] == "classification":
        train_class = mapfile_df["class"]
    else:
        ValueError(f"Invalid value for params.task:\t{params['task']}\nExpected ['classification','segmentation']")

    # K-fold split into train and dev sets stratified by train_labels
    # using random seed for reproducibility
    skf = StratifiedKFold(
        n_splits=params_split["n_split"],
        random_state=params_split["random_seed"],
        shuffle=params_split["shuffle"],
    )

    # create splits stratified by labels in train_class
    split_df = pd.DataFrame()
    for n_fold, (train_idx, test_idx) in enumerate(skf.split(train_files, train_class)):
        fold_name = f"fold_{n_fold + 1:02d}"

        # create intermediate dataframe for each fold
        temp_df = pd.DataFrame({"image_id": train_idx, fold_name: "train"}).set_index(
            "image_id"
        )
        temp_df = pd.concat(
            [
                temp_df,
                pd.DataFrame({"image_id": test_idx, fold_name: "test"}).set_index(
                    "image_id"
                ),
            ],
            axis=0,
            join="outer",
        )

        # append first fold to empty dataframe or join cols if n_fold > 0
        split_df = (
            pd.concat([split_df, temp_df], axis=0, join="outer")
            if n_fold == 0
            else split_df.join(temp_df)
        )

    # sort by index
    split_df = split_df.sort_index()

    if output_dir:
        assert os.path.isdir(output_dir), NotADirectoryError
        output_dir = Path(output_dir).resolve()

        # save output dataframe with indices for train and dev sets
        split_df.to_csv(output_dir.joinpath(output_filename), na_rep="nan")

    return split_df
