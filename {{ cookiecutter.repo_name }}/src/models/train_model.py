#!/usr/bin/env python3

import os
import click
from dotenv import find_dotenv, load_dotenv
import logging
from pathlib import Path

from functools import partial

import tensorflow as tf
from tensorflow.data import AUTOTUNE

import numpy as np
import cv2


# load custom libraries from src
from src.data import mapfile, load_data
from src.img.tfreader import tf_imread, tf_imreadpair, tf_dataset
from src.img.augment import apply_transforms, apply_transforms_pair
from src.data import load_params
from src.models import networks

# set tf warning options
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def train(
    mapfile_path,
    cv_idx_path,
    params_filepath="params.yaml",
    model_dir="./models",
    model_name="model",
    debug=False,
):
    """Accept filepaths to the mapfile and train-dev split, train mode, and return
    training history."""
    assert type(mapfile_path) is str, TypeError(f"MAPFILE must be type STR")

    # start logger
    logger = logging.getLogger(__name__)

    # read files
    mapfile_df, cv_idx = load_data(
        [mapfile_path, cv_idx_path], sep=",", header=0, index_col=0, dtype=str
    )

    # load params
    params = load_params(params_filepath)
    train_params = params["train_model"]
    random_seed, target_size, n_classes, mean_img, std_img = (
        params["random_seed"],
        params["target_size"],
        params["n_classes"],
        params["mean_img"],
        params["std_img"],
    )
    batch_size, epochs = train_params["batch_size"], train_params["epochs"]

    # set random seed
    tf.random.set_seed(random_seed)

    # create dataset using tf.data
    data_records = [list(elem) for elem in mapfile_df.to_records(index=False)]
    dataset = tf.data.Dataset.from_tensor_slices(data_records)

    # build tf.data pipeline
    if params["segmentation"]:
        dataset = dataset.map(
            tf_imreadpair,
            num_parallel_calls=AUTOTUNE,
        )
        if not debug:
            dataset = dataset.map(
                lambda x, y: apply_transforms_pair(
                    x, y, input_shape=target_size, mean=mean_img, std=std_img
                ),
                num_parallel_calls=AUTOTUNE,
            )
        else:
            logger.info(f"Debug mode activated. No data augmentation or shuffling")
    else:
        dataset = dataset.map(
            tf_imread,
            num_parallel_calls=AUTOTUNE,
        )
        # apply transforms except while debugging
        if not debug:
            dataset = dataset.map(
                lambda x, y: apply_transforms(x, y, input_shape=target_size),
                num_parallel_calls=AUTOTUNE,
            )
        else:
            logger.info(f"Debug mode activated. No data augmentation or shuffling")

    # # apply transforms except while debugging
    # if not debug:
    #     dataset = dataset.map(
    #         lambda x, y: apply_transforms_pair(x, y, input_shape=target_size),
    #         num_parallel_calls=AUTOTUNE,
    #     )

    dataset = (
        dataset.cache()  # cache after mapping
        .shuffle(  # shuffle after caching to randomize order TODO - turn off with debug
            buffer_size=mapfile_df.shape[0],
            seed=random_seed,
            reshuffle_each_iteration=True,
        )
        .repeat()  # infinite cardinality
        .batch(
            batch_size=batch_size,
            drop_remainder=True,
            num_parallel_calls=AUTOTUNE,
            deterministic=debug,
        )
        .prefetch(AUTOTUNE)
    )

    # create model
    if params["segmentation"]:
        model = networks.unet_xception(input_shape=target_size, n_classes=n_classes)
        optimizer = tf.keras.optimizers.Adam(0.001)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        # print(model.summary())
    else:
        model = networks.simple_nn(
            input_shape=target_size, n_classes=n_classes, debug=debug
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    # train model
    logger.info(f"Training model for {epochs} epochs")
    history = model.fit(
        dataset, steps_per_epoch=mapfile_df.shape[0], epochs=epochs, verbose=1
    )

    # save model
    model_filename = Path(model_dir).joinpath(f"{model_name}_{epochs:03d}")
    # results_filename = Path(results_dir).joinpath(f"{model_name}_{epochs}")

    # norm = np.zeros(img_predict.shape)
    # img_predict2 = cv2.normalize(img_predict, norm, 0, 255, cv2.NORM_MINMAX).astype(
    #     np.uint8
    # )
    # cv2.imwrite(str(img_filepath), img_predict2)

    model.save(model_filename, save_format="tf")

    return history


@click.command()
@click.argument(
    "mapfile_path",
    default=Path("./data/interim/mapfile_df.csv").resolve(),
    type=click.Path(exists=True),
    # help="Filepath to the CSV with image filenames and class labels.",
)
@click.argument(
    "cv_idx_path",
    default=Path("./data/processed/split_train_dev.csv").resolve(),
    type=click.Path(exists=True),
    # help="Filepath to the CSV with the cross validation train/dev splits.",
)
@click.option("--params_filepath", "-p", default="params.yaml")
# @click.option(
#     "--results-dir",
#     default=Path("./results").resolve(),
#     type=click.Path(),
# )
@click.option(
    "--model-dir",
    default=Path("./models").resolve(),
    type=click.Path(),
)
@click.option(
    "--model-name",
    default="model",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Debug switch that turns off augmentation, shuffle, and makes runs deterministic.",
)
def main(
    mapfile_path,
    cv_idx_path,
    params_filepath="params.yaml",
    # results_dir="./results",
    model_dir="./models",
    model_name="model",
    debug=False,
):
    train(
        mapfile_path,
        cv_idx_path,
        params_filepath=params_filepath,
        # results_dir=results_dir,
        model_dir=model_dir,
        model_name=model_name,
        debug=debug,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
