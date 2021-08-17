#!/usr/bin/env python3

import pandas as pd
import pytest
from pathlib import Path
import tensorflow as tf

from src.img.tfreader import tf_imread, tf_dataset


@pytest.fixture
def mapfile_df():
    filepath = "./src/tests/test_data/mnist_small/pytest_mapfile.csv"
    return pd.read_csv(Path(filepath), index_col=0, dtype=str)


class TestReader:
    def test_tf_imread(self, mapfile_df):
        """ """
        # pass
        data_records = tf_dataset(mapfile_df)
        assert type(data_records) is type(tf.data.Dataset.from_tensor_slices([1]))

        img, label = tf_imread(next(iter(data_records)))
        assert img.shape == [28, 28, 1]
        assert label.numpy() == 0
