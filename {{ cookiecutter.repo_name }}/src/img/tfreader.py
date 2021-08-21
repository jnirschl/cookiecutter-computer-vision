#!/usr/bin/env python3
import pandas as pd
import tensorflow as tf


@tf.function
def tf_imread(data_records):
    # read the image from disk, decode it, resize it, and scale the
    # pixels intensities to the range [0, 1]. NOTE: 'convert_image_dtype'
    # converts between data types, automatically scaling the values
    # based on the MAX of the input dtype

    image_raw = tf.io.read_file(data_records[0])
    image = tf.image.decode_png(image_raw)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    label = tf.strings.to_number(data_records[1], tf.int32)
    return image, label


@tf.function
def tf_imreadpair(data_records):
    # read the data_records containing the file paths to two corresponding
    # images, read images from disk, decode, and scale pixels intensities
    # to the range [0, 1].

    img1 = tf.image.decode_png(tf.io.read_file(data_records[0]))
    img1 = tf.image.convert_image_dtype(img1, dtype=tf.float32)

    img2 = tf.image.decode_png(tf.io.read_file(data_records[1]))
    img2 = tf.image.convert_image_dtype(img2, dtype=tf.float32)

    return img1, img2


def tf_dataset(mapfile_df):
    """Accept a Pandas dataframe and return a TensorFlow Dataset"""
    # assert type(mapfile_df) is type(pd.DataFrame()), ValueError(
    #     f"MAPFILE_DF must be type {type(pd.Dataframe())}"
    # )

    data_records = [list(elem) for elem in mapfile_df.to_records(index=False)]
    return tf.data.Dataset.from_tensor_slices(data_records)
