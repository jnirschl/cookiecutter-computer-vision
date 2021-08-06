#!/usr/bin/env python3


import numpy as np
import pandas as pd
import pytest

from src.img import compute_mean

@pytest.fixture
def mapfile():
    return "./src/tests/test_data/pytest_mapfile.csv"


class TestTransforms():
    pass