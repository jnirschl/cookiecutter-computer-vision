#!/usr/bin/env python3

import tensorflow as tf


def simple_nn(input_shape=(28, 28), n_classes=10):
    """ """
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(n_classes),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
