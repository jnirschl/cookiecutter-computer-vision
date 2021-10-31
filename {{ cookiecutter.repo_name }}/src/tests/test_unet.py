#!/usr/bin/env python3
# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Unet model."""

import pytest
import tensorflow as tf
from src.models import networks


class UnetTest(tf.test.TestCase):
    def test_unet_resnet(self):

        rand_seed = 83922
        tf.random.set_seed(rand_seed)
        dataset_size = 10
        batch_size = 5
        input_shape = (224, 224, 3)
        num_classes = 34

        features = tf.random.normal((dataset_size,) + input_shape)
        labels = tf.ones((dataset_size,) + input_shape[:2] + (1,))
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

        model = networks.unet(
            input_shape=input_shape,
            filters=[512, 256, 128, 64],
            num_classes=num_classes,
            seed=rand_seed,
        )

        model.compile(
            "adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        )

        history = model.fit(
            dataset,
            steps_per_epoch=dataset_size // batch_size,
            epochs=2,
        )

        loss_history = history.history["loss"]
        assert loss_history  # (loss_history > 0.0).all()
        self.assertAllGreaterEqual(loss_history, 0.0)


# if __name__ == "__main__":
#     tf.test.main()
