import math
import numpy as np
from scipy.ndimage import convolve1d,gaussian_filter1d
import torch

def unbiased_rmse(y_true, y_pred):
    predmean = np.nanmean(y_pred)
    targetmean = np.nanmean(y_true)
    predanom = y_pred-predmean
    targetanom = y_true - targetmean
    return np.sqrt(np.nanmean((predanom-targetanom)**2))


def r2_score(y_true, y_pred):
    mask = y_true == y_true
    a, b = y_true[mask], y_pred[mask]
    unexplained_error = np.nansum(np.square(a-b))
    total_error = np.nansum(np.square(a - np.nanmean(a)))
    return 1. - unexplained_error/total_error

def nanunbiased_rmse(y_true, y_pred):
    predmean = np.mean(y_pred)
    targetmean = np.mean(y_true)
    predanom = y_pred-predmean
    targetanom = y_true - targetmean
    return np.sqrt(np.mean((predanom-targetanom)**2))

def _rmse(y_true,y_pred):
    predanom = y_pred
    targetanom = y_true
    return np.sqrt(np.nanmean((predanom-targetanom)**2))
def _bias(y_true,y_pred):
    bias = np.nanmean(np.abs(y_pred-y_true))
    return bias

def _ACC(y_true,y_pred):
    y_true_anom = y_ture-np.nanmean(y_ture)
    y_pred_anom = y_pred-np.nanmean(y_pred)
    numerator = np.sum(y_true_anom*y_pred_anom)
    denominator = np.sqrt(np.sum(y_true_anom**2))*np.sqrt(np.sum(y_pred_anom**2))
    acc = numerator/denominator
    return acc


def get_bin_idx(label, bins, bins_edges):
    if math.isnan(label):
        return bins - 1
    if label > 0.6:
        return bins - 1
    else:
        return np.where(bins_edges >= label)[0][0] - 1


