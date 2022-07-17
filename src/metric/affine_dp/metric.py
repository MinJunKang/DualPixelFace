# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# Need tensorflow 2.3 version
"""Computes metrics from prediction and ground truth."""
import numpy as np
import tensorflow as tf


def _get_affine_inv_wmae(prediction,
                         depth,
                         depth_conf,
                         irls_iters=5,
                         epsilon=1e-3):
    """Gets affine invariant weighted mean average error."""
    # This function returns L1 error, but does IRLS on epsilon-invariant L1 error
    # for numerical reasons.
    prediction_vec = tf.compat.v2.reshape(prediction, [-1])
    depth_conf_vec = tf.compat.v2.reshape(depth_conf, [-1])
    irls_weight = tf.compat.v2.ones_like(depth_conf_vec)
    for _ in range(irls_iters):
        sqrt_weight = tf.compat.v2.sqrt(depth_conf_vec * irls_weight)
        lhs = sqrt_weight[:, tf.compat.v2.newaxis] * tf.compat.v2.stack(
            [prediction_vec, tf.compat.v2.ones_like(prediction_vec)], 1)
        rhs = sqrt_weight * tf.compat.v2.reshape(depth, [-1])
        affine_est = tf.compat.v2.linalg.lstsq(
            lhs, rhs[:, tf.compat.v2.newaxis], l2_regularizer=1e-5, fast=False)
        prediction_affine = prediction * affine_est[0] + affine_est[1]
        resid = tf.compat.v2.abs(prediction_affine - depth)
        irls_weight = tf.compat.v2.reshape(1. / tf.compat.v2.maximum(epsilon, resid), [-1])
    wmae = tf.compat.v2.reduce_sum(depth_conf * resid) / tf.compat.v2.reduce_sum(depth_conf)
    return wmae


def _get_affine_inv_wrmse(prediction, depth, depth_conf):
    """Gets affine invariant weighted root mean squared error."""
    prediction_vec = tf.compat.v2.reshape(prediction, [-1])
    depth_conf_vec = tf.compat.v2.reshape(depth_conf, [-1])
    lhs = tf.compat.v2.sqrt(depth_conf_vec)[:, tf.compat.v2.newaxis] * tf.compat.v2.stack(
        [prediction_vec, tf.compat.v2.ones_like(prediction_vec)], 1)
    rhs = tf.compat.v2.sqrt(depth_conf_vec) * tf.compat.v2.reshape(depth, [-1])
    affine_est = tf.compat.v2.linalg.lstsq(lhs, rhs[:, tf.compat.v2.newaxis], l2_regularizer=1e-5, fast=False)
    prediction_affine = prediction * affine_est[0] + affine_est[1]
    # Clip the residuals to prevent infs.
    resid_sq = tf.compat.v2.minimum(
        (prediction_affine - depth) ** 2,
        np.finfo(np.float32).max)
    wrmse = tf.compat.v2.sqrt(
        tf.compat.v2.reduce_sum(depth_conf * resid_sq) / tf.compat.v2.reduce_sum(depth_conf))
    return wrmse


def _pearson_correlation(x, y, w):
    """Gets Pearson correlation between `x` and `y` weighted by `w`."""
    w_sum = tf.compat.v2.reduce_sum(w)
    expectation = lambda z: tf.compat.v2.reduce_sum(w * z) / w_sum
    mu_x = expectation(x)
    mu_y = expectation(y)
    var_x = expectation(x ** 2) - mu_x ** 2
    var_y = expectation(y ** 2) - mu_y ** 2
    cov = expectation(x * y) - mu_x * mu_y
    rho = cov / tf.compat.v2.math.sqrt(var_x * var_y)
    return rho


def _get_spearman_rank_correlation(x, y, w):
    """Gets weighted Spearman rank correlation coefficent between `x` and `y`."""
    x = tf.compat.v2.reshape(x, [-1])
    y = tf.compat.v2.reshape(y, [-1])
    w = tf.compat.v2.reshape(w, [-1])
    # Argsort twice returns each item's rank.
    rank = lambda z: tf.compat.v2.argsort(tf.compat.v2.argsort(z))

    # Cast and rescale the ranks to be in [-1, 1] for better numerical stability.
    def _cast_and_rescale(z):
        return tf.compat.v2.cast(z - tf.compat.v2.shape(z)[0] // 2, tf.compat.v2.float32) / (
            tf.compat.v2.cast(tf.compat.v2.shape(z)[0] // 2, tf.compat.v2.float32))

    x_rank = _cast_and_rescale(rank(x))
    x_rank_negative = _cast_and_rescale(rank(-x))

    y_rank = _cast_and_rescale(rank(y))

    # Spearman rank correlation is just pearson correlation on
    # (any affine transformation of) rank. We take maximum in order to get
    # the absolute value of the correlation coefficient.
    return tf.compat.v2.maximum(
        _pearson_correlation(x_rank, y_rank, w),
        _pearson_correlation(x_rank_negative, y_rank, w))


def metrics(prediction, gt_depth, gt_depth_conf, crop_height=None, crop_width=None):
    """Computes and returns WMAE, WRMSE and Spearman's metrics."""
    '''
    prediction : inverse depth prediction (or disparity)
    gt_depth : inverse gt depth prediction (or disparity)
    '''

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    def center_crop(image):
        height = image.shape[0]
        width = image.shape[1]
        if (crop_height is not None) and (crop_width is not None):
            offset_y = (height - crop_height) // 2
            offset_x = (width - crop_width) // 2
            end_y = offset_y + crop_height
            end_x = offset_x + crop_width
            image = image[offset_y:end_y, offset_x:end_x].astype(np.float32)
        else:
            image = image.astype(np.float32)
        return tf.compat.v2.convert_to_tensor(image)

    # input : [B, H, W] numpy array
    # output : [wmae, wrmse, spearman]
    batch_size = prediction.shape[0]

    wmae = []
    wrmse = []
    spearman = []

    for idx in range(batch_size):
        pred_part = center_crop(prediction[idx])
        gt_depth_part = center_crop(gt_depth[idx])
        gt_depth_conf_part = center_crop(gt_depth_conf[idx])

        wmae.append(_get_affine_inv_wmae(pred_part, gt_depth_part, gt_depth_conf_part))
        wrmse.append(_get_affine_inv_wrmse(pred_part, gt_depth_part, gt_depth_conf_part))
        spearman.append(1.0 - _get_spearman_rank_correlation(pred_part, gt_depth_part, gt_depth_conf_part))

    wmae = (sum(wmae) / batch_size).numpy()
    wrmse = (sum(wrmse) / batch_size).numpy()
    spearman = (sum(spearman) / batch_size).numpy()

    tf.keras.backend.clear_session()

    return [wmae, wrmse, spearman]
