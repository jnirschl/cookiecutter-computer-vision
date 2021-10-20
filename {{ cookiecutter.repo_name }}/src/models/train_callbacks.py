#!/usr/bin/env python3

import os
import click
from dotenv import find_dotenv, load_dotenv
from codetiming import Timer
import logging
from pathlib import Path

from functools import partial

import tensorflow as tf

from src.data import load_params


def set(params_filepath="params.yaml"):
    """Accept params_filepaths, loads params.yaml, and returns a list of tf callbacks."""

    # load params
    params = load_params(params_filepath)
    callback_params = params["train_model"]["callbacks"]

    # set callbacks
    callback = []
    for elem in callback_params["callbacks"]:
        if elem.lower() in ["earlystopping"]:
            callback.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=callback_params["monitor"],
                    min_delta=0,
                    patience=15,
                    verbose=0,
                    mode=callback_params["mode"],
                    baseline=None,
                    restore_best_weights=False,
                )
            )

        if elem.lower() in ["reducelronplateau"]:
            callback.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=callback_params["monitor"],
                    factor=0.1,
                    patience=5,
                    verbose=0,
                    mode=callback_params["mode"],
                    min_delta=0.0001,
                    cooldown=0,
                    min_lr=0,
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
