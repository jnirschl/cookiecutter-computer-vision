#!/usr/bin/env python3

import logging
import os
from functools import partial
from pathlib import Path

import click
import tensorflow as tf
from codetiming import Timer
from dotenv import find_dotenv, load_dotenv

from src.data import load_params


def set(params_filepath="params.yaml"):
    """Accept params_filepaths, loads params.yaml, and returns a list of tf callbacks."""

    # start logger
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # load params
    params = load_params(params_filepath)
    callback_params = params["train_model"]["callbacks"]

    # set callbacks
    callback = []
    for elem in callback_params["callbacks"]:
        if elem.lower() in ["earlystopping"]:
            logger.info(f"Callback: EarlyStopping")
            callback.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=callback_params["monitor"],
                    min_delta=0,
                    patience=15,
                    verbose=1,
                    mode=callback_params["mode"],
                    baseline=None,
                    restore_best_weights=False,
                )
            )

        if elem.lower() in ["reducelronplateau"]:
            logger.info(f"Callback: ReduceLROnPlateau")
            callback.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=callback_params["monitor"],
                    factor=0.05,
                    patience=5,
                    verbose=1,
                    mode=callback_params["mode"],
                    min_delta=0.0001,
                    cooldown=0,
                    min_lr=1e-6,
                )
            )

    return callback

    # tf.keras.callbacks.TensorBoard(
    #     log_dir="logs",
    #     histogram_freq=0,
    #     write_graph=True,
    #     write_images=False,
    #     update_freq="epoch",
    #     profile_batch=2,
    #     embeddings_freq=0,
    #     embeddings_metadata=None,
    # )
