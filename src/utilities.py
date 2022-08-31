from six.moves import reduce
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix


def recall(labels, predictions, num_classes, pos_indices=None, weights=None, average='micro'):
    """
    Multi class recall for Tensorflow
    :param labels     : (tf.int32 or tf.int64 Tensor) True labels
    :param predictions: (tf.int32 or tf.int64 Tensor) Predictions same shape as the labels
    :param num_classes: (Int) number of classes
    :param pos_indices: (List) indices of positive classes, default = all
    :param weights    : (tf.int32 Tensor) Mask, must be of compatible shape with labels, default = None
    :param average    : (String) 'micro'    -> counts the total number of true positives, false
                                               positives, and false negatives for the classes in
                                               'pos_indices' and infer the metric from it.
                                 'macro'    -> will compute the metric separately for each class in
                                               'pos_indices' and average. Will not account for class imbalance.
                                 'weighted' -> will compute the metric separately for each class in
                                               'pos_indices' and perform a weighted average by the total
                                               number of true labels for each class.
    :return: Tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, re, _ = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    _, op, _ = metrics_from_confusion_matrix(
        op, pos_indices, average=average)
    return (re, op)


def precision(labels, predictions, num_classes, pos_indices=None, weights=None, average='micro'):
    """
    Multi class precision for Tensorflow
    :param labels     : (tf.int32 or tf.int64 Tensor) True labels
    :param predictions: (tf.int32 or tf.int64 Tensor) Predictions same shape as the labels
    :param num_classes: (Int) number of classes
    :param pos_indices: (List) indices of positive classes, default = all
    :param weights    : (tf.int32 Tensor) Mask, must be of compatible shape with labels, default = None
    :param average    : (String) 'micro'    -> counts the total number of true positives, false
                                               positives, and false negatives for the classes in
                                               'pos_indices' and infer the metric from it.
                                 'macro'    -> will compute the metric separately for each class in
                                               'pos_indices' and average. Will not account for class imbalance.
                                 'weighted' -> will compute the metric separately for each class in
                                               'pos_indices' and perform a weighted average by the total
                                               number of true labels for each class.
    :return: Tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    pr, _, _ = metrics_from_confusion_matrix(
        cm, pos_indices, average=average)
    op, _, _ = metrics_from_confusion_matrix(
        op, pos_indices, average=average)
    return (pr, op)


def f1(labels, predictions, num_classes, pos_indices=None, weights=None, average='micro'):
    return fbeta(labels, predictions, num_classes, pos_indices, weights, average)


def fbeta(labels, predictions, num_classes, pos_indices=None, weights=None, average='micro', beta=1):
    """
    Multi class fbeta metric for Tensorflow
    :param labels     : (tf.int32 or tf.int64 Tensor) True labels
    :param predictions: (tf.int32 or tf.int64 Tensor) Predictions same shape as the labels
    :param num_classes: (Int) number of classes
    :param pos_indices: (List) indices of positive classes, default = all
    :param weights    : (tf.int32 Tensor) Mask, must be of compatible shape with labels, default = None
    :param average    : (String) 'micro'    -> counts the total number of true positives, false
                                               positives, and false negatives for the classes in
                                               'pos_indices' and infer the metric from it.
                                 'macro'    -> will compute the metric separately for each class in
                                               'pos_indices' and average. Will not account for class imbalance.
                                 'weighted' -> will compute the metric separately for each class in
                                               'pos_indices' and perform a weighted average by the total
                                               number of true labels for each class.
    :param beta       : (Int) Weight of precision in harmonic mean, default = 1
    :return: Tuple of (scalar float Tensor, update_op)
    """
    cm, op = _streaming_confusion_matrix(
        labels, predictions, num_classes, weights)
    _, _, fbeta = metrics_from_confusion_matrix(
        cm, pos_indices, average=average, beta=beta)
    _, _, op = metrics_from_confusion_matrix(
        op, pos_indices, average=average, beta=beta)
    return (fbeta, op)


def safe_div(numerator, denominator):
    """Safe division, return 0 if denominator is 0"""
    numerator, denominator = tf.to_float(numerator), tf.to_float(denominator)
    zeros = tf.zeros_like(numerator, dtype=numerator.dtype)
    denominator_is_zero = tf.equal(denominator, zeros)
    return tf.where(denominator_is_zero, zeros, numerator / denominator)


def pr_re_fbeta(cm, pos_indices, beta=1):
    """Uses a confusion matrix to compute precision, recall and fbeta"""
    num_classes = cm.shape[0]
    neg_indices = [i for i in range(num_classes) if i not in pos_indices]
    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, neg_indices] = 0
    diag_sum = tf.reduce_sum(tf.diag_part(cm * cm_mask))

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[:, neg_indices] = 0
    tot_pred = tf.reduce_sum(cm * cm_mask)

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, :] = 0
    tot_gold = tf.reduce_sum(cm * cm_mask)

    pr = safe_div(diag_sum, tot_pred)
    re = safe_div(diag_sum, tot_gold)
    fbeta = safe_div((1. + beta**2) * pr * re, beta**2 * pr + re)

    return pr, re, fbeta


def metrics_from_confusion_matrix(cm, pos_indices=None, average='micro', beta=1):
    """
    Precision, Recall and F1 from the confusion matrix
    :param cm         : (tf.int32 Tensor of shape <num_classes, num_classes>) Streaming confusion matrix
    :param pos_indices: (List) indices of positive classes, default = all
    :param average    : (String) 'micro'    -> counts the total number of true positives, false
                                              positives, and false negatives for the classes in
                                              'pos_indices' and infer the metric from it.
                                'macro'    -> will compute the metric separately for each class in
                                              'pos_indices' and average. Will not account for class imbalance.
                                'weighted' -> will compute the metric separately for each class in
                                              'pos_indices' and perform a weighted average by the total
                                              number of true labels for each class.
    :param beta       : (Int) Weight of precision in harmonic mean, default = 1
    :return: Tuple of (scalar float Tensor, update_op)
    """
    num_classes = cm.shape[0]
    if pos_indices is None:
        pos_indices = [i for i in range(num_classes)]

    if average == 'micro':
        return pr_re_fbeta(cm, pos_indices, beta)
    elif average in {'macro', 'weighted'}:
        precisions, recalls, fbetas, n_golds = [], [], [], []
        for idx in pos_indices:
            pr, re, fbeta = pr_re_fbeta(cm, [idx], beta)
            precisions.append(pr)
            recalls.append(re)
            fbetas.append(fbeta)
            cm_mask = np.zeros([num_classes, num_classes])
            cm_mask[idx, :] = 1
            n_golds.append(tf.to_float(tf.reduce_sum(cm * cm_mask)))

        if average == 'macro':
            pr = tf.reduce_mean(precisions)
            re = tf.reduce_mean(recalls)
            fbeta = tf.reduce_mean(fbetas)
            return pr, re, fbeta
        if average == 'weighted':
            n_gold = tf.reduce_sum(n_golds)
            pr_sum = sum(p * n for p, n in zip(precisions, n_golds))
            pr = safe_div(pr_sum, n_gold)
            re_sum = sum(r * n for r, n in zip(recalls, n_golds))
            re = safe_div(re_sum, n_gold)
            fbeta_sum = sum(f * n for f, n in zip(fbetas, n_golds))
            fbeta = safe_div(fbeta_sum, n_gold)
            return pr, re, fbeta

    else:
        raise NotImplementedError()


def masked_conv1d_and_max(t, weights, filters, kernel_size):
    """
    Applies 1d convolution and a masked max-pooling
    :param t          : (tf.Tensor)  A tensor with at least 3 dimensions [d1, d2, ..., dn-1, dn]
    :param weights    : (tf.Tensor or tf.bool) A Tensor of shape [d1, d2, dn-1]
    :param filters    : (Int) number of filters
    :param kernel_size: (Int) kernel size for the temporal convolution
    :return: (tf.Tensor) A tensor of shape [d1, d2, dn-1, filters]
    """

    # Get shape and parameters
    shape = tf.shape(t)
    ndims = t.shape.ndims
    dim1 = reduce(lambda x, y: x*y, [shape[i] for i in range(ndims - 2)])
    dim2 = shape[-2]
    dim3 = t.shape[-1]

    # Reshape weights
    weights = tf.reshape(weights, shape=[dim1, dim2, 1])
    weights = tf.to_float(weights)

    # Reshape input and apply weights
    flat_shape = [dim1, dim2, dim3]
    t = tf.reshape(t, shape=flat_shape)
    t *= weights

    # Apply convolution
    t_conv = tf.layers.conv1d(t, filters, kernel_size, padding='same')
    t_conv *= weights

    # Reduce max -- set to zero if all padded
    t_conv += (1. - weights) * tf.reduce_min(t_conv, axis=-2, keepdims=True)
    t_max = tf.reduce_max(t_conv, axis=-2)

    # Reshape the output
    final_shape = [shape[i] for i in range(ndims-2)] + [filters]
    t_max = tf.reshape(t_max, shape=final_shape)

    return t_max
