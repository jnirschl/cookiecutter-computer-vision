#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers


def unet(input_shape=(256, 256, 1), base_filter=4, n_classes=1):
    inputs = layers.Input(input_shape)
    conv1 = layers.Conv2D(
        base_filter,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)
    conv1 = layers.Conv2D(
        base_filter,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(
        base_filter * 2,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool1)
    conv2 = layers.Conv2D(
        base_filter * 2,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(
        base_filter * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool2)
    conv3 = layers.Conv2D(
        base_filter * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(
        base_filter * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool3)
    conv4 = layers.Conv2D(
        base_filter * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = layers.Conv2D(
        base_filter * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(pool4)
    conv5 = layers.Conv2D(
        base_filter * 16,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2D(
        base_filter * 8,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(layers.UpSampling2D(size=(2, 2))(drop5))
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(
        base_filter * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge6)
    conv6 = layers.Conv2D(
        base_filter * 8,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv6)

    up7 = layers.Conv2D(
        base_filter * 4,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(
        base_filter * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge7)
    conv7 = layers.Conv2D(
        base_filter * 4,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv7)

    up8 = layers.Conv2D(
        base_filter * 2,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(
        base_filter * 2,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge8)
    conv8 = layers.Conv2D(
        base_filter * 2,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv8)

    up9 = layers.Conv2D(
        base_filter,
        2,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(
        base_filter,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge9)
    conv9 = layers.Conv2D(
        base_filter,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(conv9)
    conv9 = layers.Conv2D(
        2, 3, activation="relu", padding="same", kernel_initializer="he_normal"
    )(conv9)
    conv10 = layers.Conv2D(n_classes, 1, activation="sigmoid")(conv9)

    return tf.keras.Model(inputs, conv10)


def unet_xception(input_shape, n_classes, padding="same"):
    """
    From https://keras.io/examples/vision/oxford_pets_image_segmentation/#prepare-unet-xceptionstyle-model
    """
    inputs = layers.Input(shape=input_shape)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64/2, 128/2, 256/2]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding=padding)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding=padding)(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding=padding)(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding=padding)(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # [Second half of the network: upsampling inputs]

    for filters in [256/2, 128/2, 64/2, 32/2]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding=padding)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding=padding)(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding=padding)(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(n_classes, 3, activation="softmax", padding=padding)(x)

    # tf.keras.backend.clear_session()
    return tf.keras.Model(inputs, outputs)


# def build_unet(input_shape):
#     inputs = Input(input_shape)
#
#     s1, p1 = encoder_block(inputs, 64)
#     s2, p2 = encoder_block(p1, 128)
#     s3, p3 = encoder_block(p2, 256)
#     s4, p4 = encoder_block(p3, 512)
#
#     b1 = conv_block(p4, 1024)
#
#     d1 = decoder_block(b1, s4, 512)
#     d2 = decoder_block(d1, s3, 256)
#     d3 = decoder_block(d2, s2, 128)
#     d4 = decoder_block(d3, s1, 64)
#
#     outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
#
#     model = Model(inputs, outputs, name="U-Net")
#     return model
