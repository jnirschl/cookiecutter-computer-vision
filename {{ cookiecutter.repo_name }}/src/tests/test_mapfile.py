#!/usr/bin/env python3

import os

import pandas as pd
import pytest
from pathlib import Path


from src.data import mapfile


@pytest.fixture
def input_dir():
    return "./src/tests/test_data/mnist_small_train"


@pytest.fixture
def output_dir():
    return "./src/tests/test_data"


@pytest.fixture
def output_filename():
    return "pytest_mapfile.csv"


@pytest.fixture
def mapfile_df(output_filename):
    return pd.read_csv(Path("./src/tests/test_data/").joinpath(output_filename))


class TestMapfile:  # input_dir, output_dir, output_filename
    def test_create_mapfile(self, input_dir, output_dir, output_filename):
        """Tests for mapfile.create.create_mapfile
        Test the output of mapfile.create is a Pandas DataFrame
        Tests that output filename and label_encoding.yaml exists"""

        mapfile_df = mapfile.create(input_dir, output_dir, output_filename)

        assert type(mapfile_df) is pd.DataFrame
        assert Path(output_dir).joinpath(output_filename).exists()
        assert Path(output_dir).joinpath("label_encoding.yaml").exists()

    def test_split(self, mapfile_df, output_dir):
        """Tests for mapfile.split
        Test that each image is a test for one and only one cross-
        validation fold. Also tests file saving"""

        split_df = mapfile.split(mapfile_df, output_dir)

        # each image should be a test for one and only one fold
        assert (
            split_df.apply(lambda x: x == "test").sum(axis=1).all()
        ), "The same image is a 'test' for more than one fold"

        # test output file exists
        assert Path(output_dir).joinpath("split_train_dev.csv").exists()

    def test_split_rng(self, mapfile_df, input_dir):
        """Tests for mapfile.split
        Test that repeated calls to mapfile.split return the same
        stratified k-fold based on the random seed in params.yaml"""

        split_1 = mapfile.split(mapfile_df)
        split_2 = mapfile.split(mapfile_df)
        assert (split_1 == split_2).all().all()