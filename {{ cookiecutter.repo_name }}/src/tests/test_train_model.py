#!/usr/bin/env python3
import pytest
from pathlib import Path
from click.testing import CliRunner

from src.models import train_model


@pytest.fixture
def mapfile_path():
    filepath = "./src/tests/test_data/pytest_mapfile.csv"
    return str(Path(filepath).resolve())


@pytest.fixture
def cv_idx_path():
    filepath = "./src/tests/test_data/split_train_dev.csv"
    return str(Path(filepath).resolve())


@pytest.fixture
def output_filename():
    return "pytest_mapfile.csv"


@pytest.fixture
def test_params():
    return "./src/tests/test_data/test_params.yaml"


def test_train_model(mapfile_path, cv_idx_path, test_params):
    """ """
    expected_history = {
        "loss": 0.47114235162734985,
        "sparse_categorical_accuracy": 0.9690625071525574,
    }

    runner = CliRunner()
    result = runner.invoke(
        train_model.main, [mapfile_path, cv_idx_path, "-p", test_params]
    )

    assert not result.exception
    assert result.exit_code == 0
