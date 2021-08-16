#!/usr/bin/env python3
import pandas as pd
import tensorflow as tf


@tf.function
def tf_imread(data_records):
    # read the image from disk, decode it, resize it, and scale the
    # pixels intensities to the range [0, 1]

    image = tf.io.read_file(data_records[0])
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32) / 255
    label = tf.strings.to_number(data_records[1], tf.int32)
    return image, label


def tf_dataset(mapfile_df):
    """Accept a Pandas dataframe and return a TensorFlow Dataset"""
    # assert type(mapfile_df) is type(pd.DataFrame()), ValueError(
    #     f"MAPFILE_DF must be type {type(pd.Dataframe())}"
    # )

    data_records = [list(elem) for elem in mapfile_df.to_records(index=False)]
    return tf.data.Dataset.from_tensor_slices(data_records)
