#   -*- coding: utf-8 -*-
#  Copyright (c) 2021.  Jeffrey J. Nirschl. All rights reserved.
#
#   Licensed under the MIT license. See the LICENSE.md file in the project
#   root directory for full license information.
#
#   Time-stamp: <>
#   ======================================================================

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data import load_data, load_params


def main(mapfile, output_dir=None):
    """Split data into train and dev sets"""

    if type(mapfile) is str:
        assert (os.path.isfile(mapfile)), FileNotFoundError
        # read file
        train_df = load_data(mapfile,
                             sep=",", header=0,
                             index_col=0)
    else:
        train_df = mapfile

    # set index
    train_df.index.name = "index"

    # load params
    params = load_params()
    params_split = params['train_test_split']
    params_split["random_seed"] = params["random_seed"]

    # get filenames and dependent variables (labels)
    train_labels = train_df.pop(params_split["target_class"])
    train_files = train_df

    # K-fold split into train and dev sets stratified by train_labels
    # using random seed for reproducibility
    skf = StratifiedKFold(n_splits=params_split['n_split'],
                          random_state=params_split['random_seed'],
                          shuffle=params_split['shuffle'])

    # create splits
    split_df = pd.DataFrame()
    for n_fold, (train_idx, test_idx) in enumerate(skf.split(train_files,
                                                             train_labels)):
        fold_name = f"fold_{n_fold + 1:02d}"

        # create intermediate dataframe for each fold
        temp_df = pd.DataFrame({"image_id": train_idx,
                                fold_name: "train"}).set_index("image_id")
        temp_df = temp_df.append(pd.DataFrame({"image_id": test_idx,
                                               fold_name: "test"}).set_index("image_id"))

        # append first fold to empty dataframe or join cols if n_fold > 0
        split_df = split_df.append(temp_df) if n_fold == 0 else split_df.join(temp_df)

    # sort by index
    split_df = split_df.sort_index()

    if output_dir:
        assert (os.path.isdir(output_dir)), NotADirectoryError
        output_dir = Path(output_dir).resolve()

        # save output dataframe with indices for train and dev sets
        split_df.to_csv(output_dir.joinpath("split_train_dev.csv"),
                        na_rep="nan")
    else:
        return split_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-tr", "--train", dest="train_path",
                        required=True, help="Train CSV file")
    parser.add_argument("-o", "--out-dir", dest="output_dir",
                        default=Path("./data/processed ").resolve(),
                        required=False, help="output directory")
    args = parser.parse_args()

    # split data into train and dev sets
    main(args.train_path, args.output_dir)
