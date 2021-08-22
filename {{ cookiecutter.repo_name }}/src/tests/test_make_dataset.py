#!/usr/bin/env python3
import os.path

import pandas as pd
import pytest
from pathlib import Path
from click.testing import CliRunner

from src.data import make_dataset


@pytest.fixture
def input_dir():
    return "./src/tests/test_data/mnist_small"


@pytest.fixture
def output_dir():
    return "./src/tests/test_data/mnist_small"


@pytest.fixture
def output_filename():
    return "pytest_mapfile.csv"


@pytest.fixture
def test_params():
    return "./src/tests/test_data/test_params.yaml"


class TestMakeDataset:
    def test_mnist_python(self, input_dir, output_dir):
        """ """
        mapfile_df = make_dataset.create(input_dir, output_dir)
        assert type(mapfile_df) is type(pd.DataFrame())

    def test_mnist_click(self, input_dir, output_dir, output_filename, test_params):
        """TODO"""
        # delete existing file
        if Path(output_dir).joinpath(output_filename).exists():
            os.remove(Path(output_dir).joinpath(output_filename))

        runner = CliRunner()
        result = runner.invoke(
            make_dataset.main,
            [
                input_dir,
                output_dir,
                output_filename,
                "-p",
                "./src/tests/test_data/test_params.yaml",
                "--force",
            ],
        )

        assert result.exit_code == 0
        assert not result.exception
        # assert (
        #     result.output.strip()
        #     == "Found 100 images belonging to 10 classes.\nProcessed 100 images."
        # )
        assert Path(output_dir).joinpath(output_filename).exists()
        assert Path(output_dir).joinpath("label_encoding.yaml").exists()
        assert Path(output_dir).joinpath("split_train_dev.csv").exists()

    def test_mito_seg_python(self, test_params):
        """ """
        input_dir = "./src/tests/test_data/mito_seg"
        output_dir = input_dir
        mapfile_df = make_dataset.create(
            input_dir,
            output_dir,
            params_filepath=str(Path(input_dir).joinpath("params.yaml").resolve()),
        )
        assert type(mapfile_df) is type(pd.DataFrame())
        assert Path(mapfile_df["filename"][0]).name == Path(mapfile_df["class"][0]).name

    def test_flowers_python(self):
        """ """
        pass
        # , input_dir, output_dir, output_filename
        # mapfile_df = make_dataset.create(input_dir, output_dir, output_filename)
        # assert type(mapfile_df) is type(pd.DataFrame())

    # def test_flowers_click(self):
    #     """TODO"""
    #     input_dir = Path(
    #         "/home/jeff/Documents/GitHub/cookiecutter-computer-vision/{{ cookiecutter.repo_name }}/src/tests/test_data/flowers"
    #     )
    #     output_dir = input_dir
    #     output_filename = "mapfile_df"
    #
    #     runner = CliRunner()
    #     result = runner.invoke(
    #         make_dataset.main,
    #         [
    #             input_dir,
    #             output_dir,
    #             output_filename,
    #             "-p",
    #             output_dir.joinpath("params.yaml"),
    #             "--force",
    #         ],
    #     )
    #
    #     assert result.exit_code == 0
