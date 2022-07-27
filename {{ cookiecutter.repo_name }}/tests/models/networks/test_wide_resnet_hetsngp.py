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

"""Tests for Wide HetSNGP ResNet."""

import tensorflow as tf

from src.models import networks


class WideResnetHetSNGPTest(tf.test.TestCase):
    def testWideResnetHetSNGP(self):
        tf.random.set_seed(83922)
        dataset_size = 10
        batch_size = 5
        input_shape = (32, 32, 1)
        num_classes = 3

        features = tf.random.normal((dataset_size,) + input_shape)
        coeffs = tf.random.normal([tf.reduce_prod(input_shape), num_classes])
        net = tf.reshape(features, [dataset_size, -1])
        logits = tf.matmul(net, coeffs)
        labels = tf.random.categorical(logits, 1)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

        model = networks.wide_resnet_hetsngp(
            input_shape=input_shape,
            batch_size=batch_size,
            depth=10,
            width_multiplier=1,
            num_classes=num_classes,
            l2=0.0,
            use_mc_dropout=False,
            use_filterwise_dropout=False,
            dropout_rate=0.0,
            use_gp_layer=True,
            gp_input_dim=128,
            gp_hidden_dim=1024,
            gp_scale=1.0,
            gp_bias=0.0,
            gp_input_normalization=False,
            gp_random_feature_type="orf",
            gp_cov_discount_factor=-1.0,
            gp_cov_ridge_penalty=1.0,
            use_spec_norm=True,
            spec_norm_iteration=1,
            spec_norm_bound=6,
            temperature=1.0,
            num_factors=3,
            num_mc_samples=1000,
            eps=1e-5,
            sngp_var_weight=1.0,
            het_var_weight=1.0,
        )

        model.compile(
            "adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        history = model.fit(
            dataset, steps_per_epoch=dataset_size // batch_size, epochs=2
        )

        loss_history = history.history["loss"]
        self.assertAllGreaterEqual(loss_history, 0.0)


if __name__ == "__main__":
    tf.test.main()
