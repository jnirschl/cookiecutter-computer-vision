import tensorflow as tf
from tensorflow.keras import layers


def unet_xception(
    input_shape: tuple,
    num_classes: int,
    padding: str = "same",
    downsample: int = 2,
) -> tf.keras.models.Model:
    """
    From https://keras.io/examples/vision/oxford_pets_image_segmentation/#prepare-unet-xceptionstyle-model
    """
    inputs = layers.Input(shape=input_shape)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32 / downsample, 3, strides=2, padding=padding)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    filter_block_1 = [int(elem / downsample) for elem in [64, 128, 256]]
    filter_block_2 = [int(elem / downsample) for elem in [256, 128, 64, 32]]
    for filters in filter_block_1:
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

    for filters in filter_block_2:
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
    outputs = layers.Conv2D(num_classes, 3, activation=None, padding=padding)(x)

    # tf.keras.backend.clear_session()
    return tf.keras.Model(inputs, outputs, name="unet_xception")


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
