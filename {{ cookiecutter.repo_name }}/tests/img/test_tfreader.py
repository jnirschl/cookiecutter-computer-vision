from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from src.img.tfreader import tf_dataset, tf_imread, tf_imreadpair


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
        img_min, img_max = np.min(img.numpy()), np.max(img.numpy())
        assert img.shape == [28, 28, 1]
        assert img.dtype == tf.float32, "Image dtype must be tf.float32"
        assert (img_min >= 0) and (img_max <= 1), "Image values must be in [0, 1]"
        assert (img_max - img_min) == 1.0, "Image range"
        assert label.numpy() == 0
        assert label.dtype == tf.int32, "Label dtype must be tf.float32"

    def test_tf_imreadpair(self, mapfile_df):
        """ """
        mapfile_df["class"] = mapfile_df["filename"]
        data_records = tf_dataset(mapfile_df)
        img, img_2 = tf_imreadpair(next(iter(data_records)))
        img_min, img_max = np.min(img.numpy()), np.max(img.numpy())
        assert (img.numpy() == img_2.numpy()).all()
        assert img.shape == [28, 28, 1]
        assert img.dtype == tf.float32, "Image dtype must be tf.float32"
        assert (img_min >= 0) and (img_max <= 1), "Image values must be in [0, 1]"
        assert (img_max - img_min) == 1.0, "Image range"
