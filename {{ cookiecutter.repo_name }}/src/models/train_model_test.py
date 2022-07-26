#!/usr/bin/env python3
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.models import train


@pytest.fixture
def mapfile_path():
    filepath = "./src/tests/test_data/mnist_small/pytest_mapfile.csv"
    return str(Path(filepath).resolve())


@pytest.fixture
def mapfile_path_seg():
    filepath = "./src/tests/test_data/mito_seg/pytest_mapfile.csv"
    return str(Path(filepath).resolve())


@pytest.fixture
def cv_idx_path():
    filepath = "./src/tests/test_data/mnist_small/split_train_dev.csv"
    return str(Path(filepath).resolve())


@pytest.fixture
def cv_idx_path_seg():
    filepath = "./src/tests/test_data/mito_seg/split_train_dev.csv"
    return str(Path(filepath).resolve())


@pytest.fixture
def output_filename():
    return "pytest_mapfile.csv"


@pytest.fixture
def mnist_params():
    return "./src/tests/test_data/mnist_small/params.yaml"


@pytest.fixture
def mito_seg_params():
    return "./src/tests/test_data/mito_seg/params.yaml"


class TestTrainModel:
    def test_mnist_python(self, mapfile_path, cv_idx_path, mnist_params):
        """ """
        expected_history = {
            "loss": 2.7250454425811768,
            "accuracy": 0.02901785634458065,
        }

        history = train.fit(
            mapfile_path, cv_idx_path, params_filepath=mnist_params, debug=True
        )
        # assert abs(history.history["loss"][-1] - expected_history["loss"]) < 0.0001
        # assert (
        #     abs(history.history["accuracy"][-1] - expected_history["accuracy"]) < 0.0001
        # )

    def test_mnist_click(self, mapfile_path, cv_idx_path, mnist_params):
        """ """

        runner = CliRunner()
        result = runner.invoke(
            train.main, [mapfile_path, cv_idx_path, "-p", mnist_params, "-d"]
        )

        assert not result.exception
        assert result.exit_code == 0

    def test_mito_seg(self, mapfile_path_seg, cv_idx_path_seg, mito_seg_params):
        """ """
        pass
        # history = train_model.train(
        #     mapfile_path_seg,
        #     cv_idx_path_seg,
        #     params_filepath=test_params_seg,
        #     debug=True,
        # )
        # assert history.history["loss"][-1] == expected_history["loss"]
        # assert (
        #     history.history["sparse_categorical_accuracy"][-1]
        #     == expected_history["sparse_categorical_accuracy"]
        # )

    def test_mito_seg_click(self, mapfile_path_seg, cv_idx_path_seg, mito_seg_params):
        """ """

        pass
        # runner = CliRunner()
        # result = runner.invoke(
        #     train_model.main, [mapfile_path_seg, cv_idx_path_seg, "-p", test_params]
        # )
        #
        # assert not result.exception
        # assert result.exit_code == 0
