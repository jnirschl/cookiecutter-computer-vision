import logging

import pandas as pd
import tensorflow as tf

from src.img.augment import apply_transforms, apply_transforms_pair
from src.img.tfreader import tf_imread, tf_imreadpair


def create_dataset(
    mapfile_df: pd.DataFrame,
    params: dict(),
    logger: logging._loggerClass,
) -> tf.data.Dataset:
    """Create Tensorflow Dataset instance from mapfile DataFrame and parameters

    Accepts mapfile_df as Pandas DataFrame, dictionary of parameters, and logger and returns
    a Tensorflow Dataset instance

    Args:
        mapfile_df: pd.DataFrame with full path to input directory
        params: dict containing parameters
        logger: logger class for loggins

    Returns:
        dataset: tf.Data.Dataset with input pipeline
    """

    # create dataset using tf.data
    data_records = [list(elem) for elem in mapfile_df.to_records(index=False)]
    dataset = tf.data.Dataset.from_tensor_slices(data_records)

    # set params
    batch_size = params["train_model"]["batch_size"]
    random_seed, target_size, mean_img, std_img, deterministic = (
        params["random_seed"],
        params["target_size"],
        params["mean_img"],
        params["std_img"],
        params["deterministic"],
    )

    # build tf.data pipeline - map transforms before caching
    if params["task"] == "segmentation":
        dataset = dataset.map(
            tf_imreadpair,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        if not deterministic:
            dataset = dataset.map(
                lambda x, y: apply_transforms_pair(
                    x, y, input_shape=target_size, mean=mean_img, std=std_img
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    elif params["task"] == "classification":
        dataset = dataset.map(
            tf_imread,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        # apply transforms except under deterministic mode
        if not deterministic:
            dataset = dataset.map(
                lambda x, y: apply_transforms(
                    x, y, input_shape=target_size, mean=mean_img, std=std_img
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
    else:
        ValueError(f"Invalid value for params.task:\t{params['task']}\nExpected ['classification','segmentation']")

    # cache before shuffle to randomize order
    dataset = dataset.cache()
    if not deterministic:
        dataset = dataset.shuffle(
            buffer_size=mapfile_df.shape[0],
            seed=random_seed,
            reshuffle_each_iteration=True,
        )

    # apply repeat after shuffle to show every example of one epoch before moving to the next
    # otherwise, a repeat before shuffle will mix epoch boundaries together
    dataset = (
        dataset.repeat()  # infinite cardinality
        .batch(
            batch_size=batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=deterministic,
        )
        .prefetch(tf.data.AUTOTUNE)
    )

    return dataset
