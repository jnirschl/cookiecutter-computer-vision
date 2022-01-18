#!/usr/bin/env python3

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def watershed_wrapper(mask, min_distance=20, structure=np.ones((3, 3))):
    """Watershed segmentation"""

    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    Dist = ndimage.distance_transform_edt(mask)
    local_max_coords = peak_local_max(Dist, min_distance=min_distance, labels=mask)
    local_max_mask = np.zeros(Dist.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(local_max_mask, structure=structure)[0]
    labels = watershed(-Dist, markers, mask=mask)
    return labels
