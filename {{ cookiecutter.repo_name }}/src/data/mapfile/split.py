#!/usr/bin/env python3

import os
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import load_data, load_params


def split(
    mapfile_df,
    output_dir=None,
    output_filename="split_train_dev.csv",
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

    # get filenames and dependent variables (class)
    train_class = mapfile_df["class"]
    train_files = mapfile_df["filename"]

    # load params
    params = load_params()
    params_split = params["train_test_split"]
    params_split["random_seed"] = params["random_seed"]

    # K-fold split into train and dev sets stratified by train_labels
    # using random seed for reproducibility
    skf = StratifiedKFold(
        n_splits=params_split["n_split"],
        random_state=params_split["random_seed"],
        shuffle=params_split["shuffle"],
    )

    # create splits
    split_df = pd.DataFrame()
    for n_fold, (train_idx, test_idx) in enumerate(skf.split(train_files, train_class)):
        fold_name = f"fold_{n_fold + 1:02d}"

        # create intermediate dataframe for each fold
        temp_df = pd.DataFrame({"image_id": train_idx, fold_name: "train"}).set_index(
            "image_id"
        )
        temp_df = temp_df.append(
            pd.DataFrame({"image_id": test_idx, fold_name: "test"}).set_index(
                "image_id"
            )
        )

        # append first fold to empty dataframe or join cols if n_fold > 0
        split_df = split_df.append(temp_df) if n_fold == 0 else split_df.join(temp_df)

    # sort by index
    split_df = split_df.sort_index()

    if output_dir:
        assert os.path.isdir(output_dir), NotADirectoryError
        output_dir = Path(output_dir).resolve()

        # save output dataframe with indices for train and dev sets
        split_df.to_csv(output_dir.joinpath(output_filename), na_rep="nan")

    return split_df
