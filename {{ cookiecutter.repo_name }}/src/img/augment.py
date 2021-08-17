#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import image
import tensorflow_addons as tfa

# from src.img import tf_normalize


@tf.function
def tf_normalize(img):
    """Accept a uint8 input image as a tf.data.tensor_slice and return the
    image normalized to the range [-1, 1]"""
    return tf.math.subtract(tf.math.divide(img, 127.5), 1, name="tf_normalize")


@tf.function
def tf_resize(img, height, width, resize_method=image.ResizeMethod.NEAREST_NEIGHBOR):
    """Accepts and image as tf.data.tensor_slices and returns
    a"""
    return tf.image.resize(img, [height, width], method=resize_method)


@tf.function
def random_crop(img, IMG_HEIGHT=224, IMG_WIDTH=224):
    """Accepts and image and label as tf.data.tensor_slices and returns
    the image and label with random cropping applied"""
    return tf.image.random_crop(img, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])


@tf.function
def random_crop(img, IMG_HEIGHT=224, IMG_WIDTH=224):
    """Accepts and image and label as tf.data.tensor_slices and returns
    the image and label with random cropping applied"""
    return tf.image.random_crop(img, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])


@tf.function
def random_flip(img):
    """Accepts and image and label as tf.data.tensor_slices and returns
    a transformed image with random horizontal and vertical flipping"""
    img = tf.image.random_flip_up_down(img)
    return tf.image.random_flip_left_right(img)


@tf.function
def random_brightness(img, max_delta=0.2):
    """Accepts and image and label as tf.data.tensor_slices and returns
    a transformed image with random horizontal and vertical flipping"""
    return tf.image.random_brightness(img, max_delta=max_delta)


@tf.function
def random_filtering(img, filter_shape=11):
    return tfa.image.mean_filter2d(img, filter_shape=filter_shape)


@tf.function
def random_rotate(img, rotation):
    return tfa.image.rotate(img, tf.constant(np.pi / 8))


@tf.function()
def apply_transforms(img, label, max_delta=0.2):
    # resizing to 286x286
    img = tf_resize(img, 286, 286)

    # random brightness
    img = random_brightness(img, max_delta=max_delta)

    img = random_crop(img)  # random crop

    if tf.random.uniform(()) > 0.5:
        img = random_flip(img)  # random mirror

    return img, label
