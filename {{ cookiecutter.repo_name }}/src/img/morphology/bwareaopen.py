#!/usr/bin/env python3

import cv2

# Specify opencv optimization
cv2.setUseOptimized(True)


def bwareaopen(mask, min_size, connectivity=8):
    """Remove small objects from binary image (approximation of
    bwareaopen in Matlab for 2D images).

    Args:
        img: a binary image (dtype=uint8) to remove small objects from
        min_size: minimum size (in pixels) for an object to remain in the image
        connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).

    Returns:
        the binary image with small objects removed
    """

    # Find all connected components (called here "labels")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )

    # check size of all connected components (area in pixels)
    for i in range(num_labels):
        label_size = stats[i, cv2.CC_STAT_AREA]

        # remove connected components smaller than min_size
        if label_size < min_size:
            mask[labels == i] = 0

    return mask
