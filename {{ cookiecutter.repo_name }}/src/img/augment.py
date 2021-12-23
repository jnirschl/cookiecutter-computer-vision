#!/usr/bin/env python3

import albumentations as Alb
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import image

# from src.img import tf_normalize


def tf_normalize(img, mean=[0.5, 0.5, 0.5]):
    """Accept a tf.float32 input image as a tf.data.tensor_slice and return the
    image normalized to the range [-1, 1]"""
    return tf.math.subtract(
        tf.math.divide(img, tf.constant(mean)), 1, name="tf_normalize"
    )


def tf_standardize(img, mean=[0.5, 0.5, 0.5], std=[0.11, 0.11, 0.11]):
    """Accept a tf.float32 input image as a tf.data.tensor_slice and return the
    image normalized to the range [-1, 1]"""
    return tf.math.divide(
        tf.math.subtract(img, tf.constant(mean), name="tf_standardize"),
        tf.constant(std),
    )


def tf_resize(img, height, width, resize_method=image.ResizeMethod.BILINEAR):
    """Accepts and image as tf.data.tensor_slices and returns
    a"""
    return tf.image.resize(img, [height, width], method=resize_method)


def tf_resize_pair(img, mask, height, width, resize_method=image.ResizeMethod.BILINEAR):
    """Accepts and pair of images as tf.data.tensor_slices and returns the
    resized pairs"""
    img = tf_resize(img, height, width, resize_method=resize_method)
    mask = tf_resize(mask, height, width, resize_method=resize_method)
    return img, mask


def random_crop(img, IMG_HEIGHT=224, IMG_WIDTH=224, IMG_CH=3):
    """Accept and image as tf.data.tensor_slices and returns
    the image with random cropping applied"""
    return tf.image.random_crop(img, size=[ IMG_HEIGHT, IMG_WIDTH, IMG_CH])


def random_crop_pair(img, mask, IMG_HEIGHT=224, IMG_WIDTH=224, IMG_CH=3):
    """Accept and image and mask pair as tf.data.tensor_slices and return
    the image pair with identical random cropping applied"""
    mask = tf.concat([mask] * IMG_CH, axis=2)

    stack_imgs = tf.stack([img, mask], axis=0)

    # random flip up-down
    if tf.random.uniform(()) > 0.5:
        stack_imgs = tf.image.flip_up_down(stack_imgs)

    if tf.random.uniform(()) > 0.5:
        stack_imgs = tf.image.flip_left_right(stack_imgs)

    # random rotate
    stack_imgs = tfa.image.rotate(
        stack_imgs,
        tf.constant(np.pi / np.random.randint(1, 8)),
        interpolation="nearest",
        fill_mode="reflect",
        name="random_rotate",
    )
    crop_stack = tf.image.random_crop(
        stack_imgs, size=[2, IMG_HEIGHT, IMG_WIDTH, IMG_CH], name="random_crop_pair"
    )

    # set output
    img_out = crop_stack[0]
    mask_out = crop_stack[1][:, :, 0:1]

    return img_out, mask_out


def random_flip(img):
    """Accepts and image and label as tf.data.tensor_slices and returns
    a transformed image with random horizontal and vertical flipping"""
    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_flip_up_down(img)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.random_flip_left_right(img)
    return img


# def random_brightness(img, max_delta=0.2):
#     """Accepts and image and label as tf.data.tensor_slices and returns
#     a transformed image with random horizontal and vertical flipping"""
#     return tf.image.random_brightness(img, max_delta=max_delta)


@tf.function()
def apply_transforms(img, label, input_shape, mean, std, max_delta=0.2):
    # resizing to [orig, orig*1.25]
    new_size = np.random.randint(
        input_shape[0], np.round(input_shape[0] * 1.25).astype(np.int)
    )
    img = tf_resize(img, height=new_size, width=new_size)

    # random flip up-down
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)

    # random rotate
    img = tfa.image.rotate(
        img,
        tf.constant(np.pi / np.random.randint(1, 8)),
        interpolation="nearest",
        fill_mode="reflect",
        name="random_rotate",
    )

    # random crop
    img = random_crop(
        img,
        IMG_HEIGHT=input_shape[0],
        IMG_WIDTH=input_shape[1],
        IMG_CH=input_shape[2],
    )

    # random brightness, contrast
    img = tf.image.random_brightness(img, max_delta=max_delta)
    img = tf.image.random_contrast(img, 0.8, 1.5)

    # random hue and saturation for rgb images
    if input_shape[2] > 1:
        img = tf.image.random_hue(img, 0.1)
        img = tf.image.random_saturation(img, 0.8, 1.2)

    # random jpg compression
    img = tf.image.random_jpeg_quality(img, 50, 90)

    # add random Gaussian noise
    noise = tf.random.normal(
        shape=tf.shape(img), mean=0.0, stddev=0.1, dtype=tf.float32
    )
    img = tf.add(img, noise)

    # random filtering
    img = tfa.image.mean_filter2d(img, filter_shape=np.random.randint(1, 9))

    # normalize image to range [-1, 1]
    img = tf_standardize(img, mean=mean, std=std)

    return img, label


@tf.function()
def apply_transforms_pair(img, mask, input_shape, mean, std, max_delta=0.2):
    # resizing to [orig, orig*1.25]
    new_size = np.random.randint(
        input_shape[0], np.round(input_shape[0] * 1.25).astype(np.int)
    )
    img, mask = tf_resize_pair(img, mask, new_size, new_size)

    # random crop
    img, mask = random_crop_pair(
        img,
        mask,
        IMG_HEIGHT=input_shape[0],
        IMG_WIDTH=input_shape[1],
        IMG_CH=input_shape[2],
    )

    # random brightness, contrast
    img = tf.image.random_brightness(img, max_delta=max_delta)
    img = tf.image.random_contrast(img, 0.8, 1.5)

    # random hue and saturation for rgb images
    if input_shape[2] > 1:
        img = tf.image.random_hue(img, 0.1)
        img = tf.image.random_saturation(img, 0.8, 1.2)

    # random jpg compression
    img = tf.image.random_jpeg_quality(img, 50, 90)

    # add random Gaussian noise
    noise = tf.random.normal(
        shape=tf.shape(img), mean=0.0, stddev=0.1, dtype=tf.float32
    )
    img = tf.add(img, noise)

    # random filtering
    img = tfa.image.mean_filter2d(img, filter_shape=np.random.randint(1, 9))

    # normalize image to range [-1, 1]
    img = tf_standardize(img, mean=mean, std=std)

    return img, mask


# def aug_fn(image, img_size):
#     data = {"image":image}
#     aug_data = transforms(**data)
#     aug_img = aug_data["image"]
#     aug_img = tf.cast(aug_img/255.0, tf.float32)
#     aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
#     return aug_img
#
# def process_data(image, label, img_size):
#     aug_img = tf.numpy_function(func=aug_fn, inp=[image, img_size], Tout=tf.float32)
#     return aug_img, label

# transform = Alb.Compose([
#     Alb.RandomCrop(width=256, height=256),
#     Alb.HorizontalFlip(p=0.5),
#     Alb.RandomBrightnessContrast(p=0.2),
# ])
