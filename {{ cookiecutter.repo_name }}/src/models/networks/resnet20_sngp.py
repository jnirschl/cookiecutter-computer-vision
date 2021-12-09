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

"""TF Keras definition for Resnet-20 for CIFAR."""

from typing import Any, Dict, Iterable

import functools

import edward2 as ed
import tensorflow as tf


def _resnet_layer(
    inputs: tf.Tensor,
    num_filters: int = 16,
    kernel_size: int = 3,
    strides: int = 1,
    use_activation: bool = True,
    use_norm: bool = True,
    l2_weight: float = 1e-4,
) -> tf.Tensor:
    """2D Convolution-Batch Normalization-Activation stack builder.

    Args:
      inputs: input tensor from input image or previous layer.
      num_filters: Conv2D number of filters.
      kernel_size: Conv2D square kernel dimensions.
      strides: Conv2D square stride dimensions.
      use_activation: whether or not to use a non-linearity.
      use_norm: whether to include normalization.
      l2_weight: the L2 regularization coefficient to use for the convolution
        kernel regularizer.

    Returns:
        Tensor output of this layer.
    """
    kernel_regularizer = None
    if l2_weight:
        kernel_regularizer = tf.keras.regularizers.l2(l2_weight)
    conv_layer = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=kernel_regularizer,
    )

    x = conv_layer(inputs)
    x = tf.keras.layers.BatchNormalization()(x) if use_norm else x
    x = tf.keras.layers.ReLU()(x) if use_activation is not None else x
    return x


def make_random_feature_initializer(random_feature_type: str = "orf"):
    # Use stddev=0.05 to replicate the default behavior of
    # tf.keras.initializer.RandomNormal.
    if random_feature_type == "orf":
        return ed.initializers.OrthogonalRandomFeatures(stddev=0.05)
    elif random_feature_type == "rff":
        return tf.keras.initializers.RandomNormal(stddev=0.05)
    else:
        return random_feature_type


def resnet20_sngp_add_last_layer(
    inputs,
    x,
    num_classes: int,
    use_gp_layer: bool,
    gp_input_dim: int = -1,
    gp_hidden_dim: int = 256,
    gp_scale: float = 1.0,
    gp_bias: float = 0.0,
    gp_input_normalization: bool = False,
    gp_random_feature_type: str = "orf",
    gp_cov_discount_factor: float = -1.0,
    gp_cov_ridge_penalty: float = 1.0,
    gp_output_imagenet_initializer: bool = True,
):
    """Builds ResNet50.

    Using strided conv, pooling, four groups of residual blocks, and pooling, the
    network maps spatial features of size 224x224 -> 112x112 -> 56x56 -> 28x28 ->
    14x14 -> 7x7 (Table 1 of He et al. (2015)).

    Args:
      inputs: inputs
      x: x
      num_classes: Number of output classes.
      use_gp_layer: Whether to use Gaussian process layer as the output layer.
      gp_hidden_dim: The hidden dimension of the GP layer, which corresponds to
        the number of random features used for the approximation.
      gp_scale: The length-scale parameter for the RBF kernel of the GP layer.
      gp_bias: The bias term for GP layer.
      gp_input_normalization: Whether to normalize the input using LayerNorm for
        GP layer. This is similar to automatic relevance determination (ARD) in
        the classic GP learning.
      gp_random_feature_type: The type of random feature to use for
        `RandomFeatureGaussianProcess`.
      gp_cov_discount_factor: The discount factor to compute the moving average of
        precision matrix.
      gp_cov_ridge_penalty: Ridge penalty parameter for GP posterior covariance.
      gp_output_imagenet_initializer: Whether to initialize GP output layer using
        Gaussian with small standard deviation (sd=0.01).

    Returns:
      tf.keras.Model.
    """
    # x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)

    if use_gp_layer:
        gp_output_initializer = None
        if gp_output_imagenet_initializer:
            # Use the same initializer as dense
            gp_output_initializer = tf.keras.initializers.RandomNormal(stddev=0.01)

        # Uses random projection to reduce the input dimension of the GP layer.
        if gp_input_dim > 0:
            x = tf.keras.layers.Dense(
                gp_input_dim,
                kernel_initializer="random_normal",
                use_bias=False,
                trainable=False,
            )(x)

        output_layer = functools.partial(
            ed.layers.RandomFeatureGaussianProcess,
            num_inducing=gp_hidden_dim,
            gp_kernel_scale=gp_scale,
            gp_output_bias=gp_bias,
            normalize_input=gp_input_normalization,
            gp_cov_momentum=gp_cov_discount_factor,
            gp_cov_ridge_penalty=gp_cov_ridge_penalty,
            scale_random_features=False,
            use_custom_random_features=True,
            custom_random_features_initializer=make_random_feature_initializer(
                gp_random_feature_type
            ),
            name=None,
            kernel_initializer=gp_output_initializer,
        )
    else:
        output_layer = functools.partial(
            tf.keras.layers.Dense,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            name="fc1000",
        )

    outputs = output_layer(num_classes)(x)[
        0
    ]  # TODO, check output of ed.layers.RandomFeatureGaussianProcess

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet20_sngp")


def resnet20_sngp(
    input_shape: Iterable[int],
    filters: int = 16,
    num_classes: int = 10,
    l2_weight: float = 0.0,
    batch_size: int = 32,
    **unused_kwargs: Dict[str, Any]
) -> tf.keras.models.Model:
    """Resnet-20 v1, takes tuple of input_shape and returns logits of shape (num_classes,)."""
    # TODO(znado): support NCHW data format.

    inputs = tf.keras.layers.Input(shape=input_shape)  # , batch_size=batch_size
    depth = 20
    num_filters = filters
    num_res_blocks = int((depth - 2) / 6)

    x = _resnet_layer(inputs=inputs, num_filters=num_filters, l2_weight=l2_weight)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = _resnet_layer(
                inputs=x, num_filters=num_filters, strides=strides, l2_weight=l2_weight
            )
            y = _resnet_layer(
                inputs=y,
                num_filters=num_filters,
                use_activation=False,
                l2_weight=l2_weight,
            )
            if stack > 0 and res_block == 0:
                x = _resnet_layer(
                    inputs=x,
                    num_filters=num_filters,
                    kernel_size=1,
                    strides=strides,
                    use_activation=False,
                    use_norm=False,
                    l2_weight=l2_weight,
                )
            x = tf.keras.layers.add([x, y])
            x = tf.keras.layers.ReLU()(x)
        num_filters *= 2

    x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
    x = tf.keras.layers.Flatten()(x)

    return resnet20_sngp_add_last_layer(inputs, x, num_classes, use_gp_layer=True)
