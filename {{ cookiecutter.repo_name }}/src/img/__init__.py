#!/usr/bin/env python3

import cv2

# Specify opencv optimization
cv2.setUseOptimized(True)


def to_float(img):
    return cv2.normalize(img.astype("float32"), None, 0.0, 1.0, cv2.NORM_MINMAX)
