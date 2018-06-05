import numpy as np
def calc_mean_IOU(pred, gt, thresh):
    # slightly modified from https://github.com/chrischoy/3D-R2N2/blob/master/lib/voxel.py
    pred_s = np.squeeze(pred[:, 1, :, :, :])
    preds_occupy = pred_s >= thresh
    diff = np.sum(np.logical_xor(preds_occupy, gt))
    intersection = np.sum(np.logical_and(preds_occupy, gt))
    union = np.sum(np.logical_or(preds_occupy, gt))
    num_fp = np.sum(np.logical_and(preds_occupy, np.logical_not(gt)))  # false positive
    num_fn = np.sum(np.logical_and(np.logical_not(preds_occupy), gt))  # false negative
    iou = float(intersection)/float(union)
    return np.array([diff, intersection, union, num_fp, num_fn, iou])