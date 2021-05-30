#!/usr/bin/env python3

import cv2
import numpy as np
import pandas as pd

# Specify opencv optimization
cv2.setUseOptimized(True)


def mean_image(img_array, img_shape=(28, 28, 1)):
    """Accept images as numpy array with images separated by rows
    and columns indicating pixel values"""
    assert type(img_array) is type(pd.DataFrame()), TypeError

    # pre-allocate mean_img
    mean_img = np.zeros(img_shape, dtype=np.float32)

    # process files
    print(f"Computing mean image...")
    for file_count, idx in enumerate(range(img_array.shape[0])):
        temp_img = np.reshape(img_array.iloc[idx].to_numpy(), img_shape)
        mean_img = cv2.accumulate(temp_img.astype(dtype=np.float32), mean_img)

        if file_count % 10000 == 0:
            print(f"\tProcessed {file_count:0d} images.")

    # divide by n_images
    mean_img = np.divide(mean_img, file_count + 1)

    return mean_img
