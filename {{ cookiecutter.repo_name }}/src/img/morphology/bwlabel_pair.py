import cv2
import numpy as np
from scipy import optimize
from scipy.spatial import distance

# Specify opencv optimization
cv2.setUseOptimized(True)


def bwlabel_pair(gt, mask, conn=8):
    """Match the label images from a pair of ground truth and predicted masks"""

    # get label matrix
    n_gt, label_gt, stats_gt, ctr_gt = cv2.connectedComponentsWithStats(
        gt, connectivity=conn
    )
    n_pred, label_pred, stats_pred, ctr_pred = cv2.connectedComponentsWithStats(
        mask, connectivity=conn
    )

    # compute distance cost  matrix
    dist_cost = distance.cdist(ctr_gt, ctr_pred)

    # normalize distance cost matrix
    dist_cost = (dist_cost - dist_cost.min()) / (dist_cost.max() - dist_cost.min())

    # compute intersection over union
    iou_cost = np.zeros((n_gt, n_pred), dtype=float)

    for m_row in range(n_gt):
        temp_gt = (label_gt == m_row).astype(np.uint8)

        for n_col in range(n_pred):
            temp_pred = (label_pred == n_col).astype(np.uint8)
            temp_intersect = np.sum(cv2.bitwise_and(temp_gt, temp_pred)[:]) + 1e-5
            temp_union = np.sum(cv2.bitwise_or(temp_gt, temp_pred)[:]) + 1e-5
            iou_cost[m_row, n_col] = -np.log(temp_intersect / temp_union)

    # normalize IoU matrix
    iou_cost = np.divide((iou_cost - iou_cost.min()), (iou_cost.max() - iou_cost.min()))

    # combine distance and IoU cost
    final_cost = np.divide(dist_cost + iou_cost, 2)

    # lap
    row_idx, col_idx = optimize.linear_sum_assignment(final_cost, maximize=False)

    # create new output label image
    out_gt = np.zeros_like(label_gt)
    out_pred = np.zeros_like(label_pred)
    for idx, (m_row, n_col) in enumerate(zip(row_idx, col_idx)):
        out_gt[label_gt == m_row] = idx
        out_pred[label_pred == n_col] = idx

    row_set = set(row_idx)
    gt_set = set(range(n_gt))
    unused_gt = gt_set.symmetric_difference(row_set)

    for idx, m_row in zip(range(len(row_idx), n_gt), unused_gt):
        out_gt[label_gt == m_row] = idx

    col_set = set(col_idx)
    pred_set = set(range(n_pred))
    unused_pred = pred_set.symmetric_difference(col_set)

    for idx, n_col in zip(range(len(col_idx), n_pred), unused_pred):
        out_pred[label_pred == n_col] = idx

    return out_gt, out_pred
