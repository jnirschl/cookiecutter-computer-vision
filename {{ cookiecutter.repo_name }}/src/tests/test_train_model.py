#!/usr/bin/env python3
import pytest
from pathlib import Path
from click.testing import CliRunner

from src.models import train_model


@pytest.fixture
def mapfile_path():
    filepath = "./src/tests/test_data/mnist_small/pytest_mapfile.csv"
    return str(Path(filepath).resolve())


@pytest.fixture
def cv_idx_path():
    filepath = "./src/tests/test_data/mnist_small/split_train_dev.csv"
    return str(Path(filepath).resolve())


@pytest.fixture
def output_filename():
    return "pytest_mapfile.csv"


@pytest.fixture
def test_params():
    return "./src/tests/test_data/test_params.yaml"


class TestTrainModel:
    def test_mnist_python(self, mapfile_path, cv_idx_path, test_params):
        """ """
        expected_history = {
            "loss": 2.2681679725646973,
            "sparse_categorical_accuracy": 0.5753124952316284,
        }

        history = train_model.train(
            mapfile_path, cv_idx_path, params_filepath=test_params, debug=True
        )
        assert history.history["loss"][-1] == expected_history["loss"]
        assert (
            history.history["sparse_categorical_accuracy"][-1]
            == expected_history["sparse_categorical_accuracy"]
        )

    def test_mnist_click(self, mapfile_path, cv_idx_path, test_params):
        """ """

        runner = CliRunner()
        result = runner.invoke(
            train_model.main, [mapfile_path, cv_idx_path, "-p", test_params, "--debug"]
        )

        assert not result.exception
        assert result.exit_code == 0
