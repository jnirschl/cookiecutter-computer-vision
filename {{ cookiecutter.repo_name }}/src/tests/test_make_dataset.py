#!/usr/bin/env python3
import os.path

import pytest
from pathlib import Path
from click.testing import CliRunner

from src.data import make_dataset


@pytest.fixture
def input_dir():
    return "./src/tests/test_data/mnist_small_train"


@pytest.fixture
def output_dir():
    return "./src/tests/test_data"


@pytest.fixture
def output_filename():
    return "pytest_mapfile.csv"


def test_create_mapfile(input_dir, output_dir, output_filename):
    if not os.path.isfile(output_filename):
        runner = CliRunner()
        result = runner.invoke(
            make_dataset.create_mapfile, [input_dir, output_dir, output_filename]
        )
        # print(result)
        assert result.exit_code == 0
        assert result.output.strip() == "Found 100 images belonging to 10 classes."
    else:
        pass
