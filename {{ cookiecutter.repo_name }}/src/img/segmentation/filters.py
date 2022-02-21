#!/usr/bin/env python3

import cv2
import numpy as np


def smooth(img, blur="gauss", kernel=5):
    """Image processing using smoothing filters"""
    if blur == "gauss":
        kernel = (kernel, kernel)
        smooth_img = cv2.GaussianBlur(img, kernel, cv2.BORDER_DEFAULT)
        # smoother_dividing = filters.rank.mean(util.img_as_ubyte(dividing),
        #                                  morphology.disk(4))
    elif blur == "mean":
        kernel = np.ones((kernel, kernel), np.float32) / np.power(kernel, 2)
        smooth_img = cv2.filter2D(img, -1, kernel)
    # elif blur == "meanshift":
    #   img = cv2.pyrMeanShiftFiltering(img, 21, 51)
    elif blur == "median":
        smooth_img = cv2.medianBlur(img, kernel)
    elif blur == "bilateral":
        smooth_img = cv2.bilateralFilter(img, kernel, 75, 75)

    return smooth_img
