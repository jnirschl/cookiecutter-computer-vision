#!/usr/bin/env python3

import cv2
import numpy as np
from scipy import optimize
from scipy.spatial import distance

# Specify opencv optimization
cv2.setUseOptimized(True)


def bwlabel_pair(gt, mask, conn=8):
    """Create corresponding label matrices from a pair of ground truth
    and predicted segmentations"""

    # get label matrix
    n_gt, label_gt, stats_gt, ctr_gt = cv2.connectedComponentsWithStats(
        gt, connectivity=conn
    )
    n_pred, label_pred, stats_pred, ctr_pred = cv2.connectedComponentsWithStats(
        mask, connectivity=conn
    )

    # compute cost matrix (distance)
    dist_cost = distance.cdist(ctr_gt, ctr_pred)

    # # compute cost area (abs(A_gt - A_pred))
    # area_gt = np.transpose(np.broadcast_to(stats_gt[:,cv2.CC_STAT_AREA],
    #                                        (num_labels_pred, num_labels_gt)))
    # area_pred = np.broadcast_to(stats_pred[:,cv2.CC_STAT_AREA],
    #                             (num_labels_gt, num_labels_pred))
    # area_diff = np.abs(np.subtract(area_gt, area_pred))

    # cost IoU

    # lap
    row_idx, col_idx = optimize.linear_sum_assignment(dist_cost, maximize=False)

    # create new output label image
    out_gt = np.zeros_like(label_gt)
    out_pred = np.zeros_like(label_pred)
    for idx, (m_row, n_col) in enumerate(zip(row_idx, col_idx)):
        out_gt[label_gt == m_row] = idx + 1
        out_pred[label_pred == n_col] = idx + 1

    # add gt labels that didn't match
    gt_set = set(range(n_gt))
    unused_gt = gt_set.symmetric_difference(set(row_idx))
    for idx, m_row in zip(range(len(row_idx) + 1, n_gt + 1), unused_gt):
        out_gt[label_gt == m_row] = idx

    # add pred labels that didn't match
    pred_set = set(range(n_pred))
    unused_pred = pred_set.symmetric_difference(set(col_idx))
    for idx, n_col in zip(range(len(col_idx) + 1, n_pred + 1), unused_pred):
        out_pred[label_pred == n_col] = idx

    return out_gt, out_pred
