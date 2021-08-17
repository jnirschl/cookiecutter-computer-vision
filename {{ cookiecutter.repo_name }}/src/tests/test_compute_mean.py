#!/usr/bin/env python3


import cv2
from click.testing import CliRunner
from pathlib import Path
import pytest

from src.img import compute_mean


@pytest.fixture
def mapfile_path():
    return str(Path("./src/tests/test_data/pytest_mapfile.csv").resolve())


@pytest.fixture
def output_dir():
    return "./src/tests/test_data"


@pytest.fixture
def ref_img():
    return cv2.imread("./src/tests/test_data/mean_image_ref.png", cv2.IMREAD_GRAYSCALE)


@pytest.fixture
def test_params():
    return "./src/tests/test_data/test_params.yaml"


@pytest.fixture
def grayscale():
    return True


@pytest.fixture
def force():
    return True


class TestComputeMean:
    # TODO test compute_mean RGB vs. grayscale vs. different sizes
    def test_mean_image(self, mapfile_path, output_dir, ref_img, grayscale, force):
        """Pytest function to test compute_mean.image"""

        mean_img = compute_mean.image(mapfile_path, grayscale=grayscale, force=force)
        assert Path(output_dir).joinpath("mean_image.png").exists()
        assert (mean_img == ref_img).all()

    def test_click_interface(self, mapfile_path):
        """ """
        runner = CliRunner()
        result = runner.invoke(
            compute_mean.main, [mapfile_path, "--grayscale", "--force"]
        )
        assert result.exit_code == 0
        assert not result.exception