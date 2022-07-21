__all__ = ["mapfile"]

import json
import os
from pathlib import Path

import pandas as pd
import yaml


def load_params(filepath="params.yaml") -> dict:
    """Helper function to load params.yaml

    Args:
        filepath (str): filename or full filepath to yaml file with parameters

    Returns:
        dict: dictionary of parameters
    """

    assert os.path.isfile(filepath), FileNotFoundError

    # read params.yaml
    with open(filepath, "r") as file:
        params = yaml.safe_load(file)

    return params


def convert_none_to_null(params):
    """Convert None values in params.yaml into null to ensure
    correct reading/writing as None type"""
    if isinstance(params, list):
        params[:] = [convert_none_to_null(elem) for elem in params]
    elif isinstance(params, dict):
        for k, v in params.items():
            params[k] = convert_none_to_null(v)
    return "null" if params is None else params


def save_params(params, filepath="params.yaml"):
    """ """
    # convert None values to null

    # save params
    new_params = yaml.safe_dump(params)

    with open(filepath, "w") as writer:
        writer.write(new_params)


def load_data(data_path, sep=",", header=0, index_col=0, dtype=0) -> object:
    """Helper function to load train and test files
     as well as optional param loading

    Args:
        data_path (str or list of str): path to csv file
        sep (str):
        index_col (str):
        header (int):

    Returns:
        object:
    """
    # if single path as str, convert to list of str
    if type(data_path) is str:
        data_path = [data_path]

    for elem in data_path:
        assert Path(elem).is_file(), FileNotFoundError(f"{elem}")

    # loop over filepath in list and read file
    output_df = [
        pd.read_csv(elem, sep=sep, header=header, index_col=index_col, dtype=dtype)
        for elem in data_path
    ]
    # if single file as input, return single df not a list
    if len(output_df) == 1:
        output_df = output_df[0]

    return output_df


def save_as_csv(
    df,
    filepath,
    output_dir,
    replace_text=".csv",
    suffix="_processed.csv",
    na_rep="nan",
    output_path=False,
):
    """Helper function to format the new filename and save output"""

    # if single path as str, convert to list of str

    if type(df) is not list:
        df = [df]

    if type(filepath) is str:
        filepath = [filepath]

    # list lengths must be equal
    assert len(df) == len(filepath), AssertionError

    for temp_df, temp_path in zip(df, filepath):
        # set output filenames
        save_fname = os.path.basename(temp_path.replace(replace_text, suffix))

        # save updated dataframes
        save_filepath = output_dir.joinpath(save_fname)
        temp_df.to_csv(save_filepath, na_rep=na_rep)
        if output_path:
            return save_filepath


def load_metrics(filepath="results/metrics.json") -> dict:
    """Helper function to load metrics.json

    Args:
        filepath (str): filename or full filepath to json file with metrics

    Returns:
        dict: dictionary of metrics
    """

    assert os.path.isfile(filepath), FileNotFoundError

    # read params.yaml
    with open(filepath, "r") as file:
        metrics = json.load(file)

    return metrics


def save_metrics(metrics, filepath="results/metrics.json") -> dict:
    """Helper function to save metrics.json

    Args:
        metrics (dict): dict containing with metrics
        filepath (str): filename or full filepath to save json file with metrics.
    """

    assert type(metrics) is type({}), TypeError("Expected metrics to be type dict")

    # read params.yaml
    with open(filepath, "w") as write_file:
        json.dump(metrics, write_file)
