import math
import numpy as np

from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# TODO need to clean this puppy up
# TODO fix things lol


def threshold(y_pred, thresh=0.5):
    return np.where(y_pred >= thresh, 1, 0)


def binary_confusion_mat(y_true, y_pred):
    if y_pred.dtype == float:
        y_pred = threshold(y_pred)
    if len(set(y_pred)) == 1:
        if y_pred[0] == 0:
            return np.array([[len(y_pred), 0], [0, 0]])
        elif y_pred[0] == 1:
            return np.array([[0, 0], [0, len(y_pred)]])
        else:
            raise ValueError("prediction is not binary [0, 1]")
    return confusion_matrix(y_true=y_true, y_pred=y_pred)


def ppv(y_true, y_pred):
    if y_pred.dtype == float:
        y_pred = threshold(y_pred)

    # ppv is undefined if there is no positive class in real and pred
    if np.array_equal(y_true, y_pred):
        if sum(y_true) == 0:
            return math.nan

    tn, fp, fn, tp = binary_confusion_mat(y_true=y_true, y_pred=y_pred).ravel()
    return tp / (tp + fp)


def npv(y_true, y_pred):
    if y_pred.dtype == float:
        y_pred = threshold(y_pred)

    # npv is undefined if no negative classes exist in real and pred
    if np.array_equal(y_true, y_pred):
        if sum(y_true) == len(y_true):
            return math.nan

    tn, fp, fn, tp = binary_confusion_mat(y_true=y_true, y_pred=y_pred).ravel()
    return tn / (tn + fn)


def sensitivity(y_true, y_pred):
    if y_pred.dtype == float:
        y_pred = threshold(y_pred)

    if np.array_equal(y_true, y_pred):
        if sum(y_true) == len(y_true):
            return 1.0
        else:
            return math.nan

    tn, fp, fn, tp = binary_confusion_mat(y_true=y_true, y_pred=y_pred).ravel()
    return tp / (tp + fn)


def specificity(y_true, y_pred):
    if y_pred.dtype == float:
        y_pred = threshold(y_pred)

    if np.array_equal(y_true, y_pred):
        if sum(y_true) == len(y_true):
            return math.nan
        else:
            return 1.0

    tn, fp, fn, tp = binary_confusion_mat(y_true=y_true, y_pred=y_pred).ravel()
    return tn / (tn + fp)


def accuracy(y_true, y_pred):
    if y_pred.dtype == float:
        y_pred = threshold(y_pred)

    if np.array_equal(y_true, y_pred):
        return 1.0

    tn, fp, fn, tp = binary_confusion_mat(y_true=y_true, y_pred=y_pred).ravel()
    return (tn + tp) / (tp + tn + fp + fn)


def balanced_accuracy(y_true, y_pred):
    if y_pred.dtype == float:
        y_pred = threshold(y_pred)

    if np.array_equal(y_true, y_pred):
        return 1.0

    tn, fp, fn, tp = binary_confusion_mat(y_true=y_true, y_pred=y_pred).ravel()
    return ((tn / (tn + fp)) + (tp / (tp + fn))) / 2


def f1(y_true, y_pred):
    if y_pred.dtype == float:
        y_pred = threshold(y_pred)

    if np.array_equal(y_true, y_pred):
        return 1.0

    tn, fp, fn, tp = binary_confusion_mat(y_true=y_true, y_pred=y_pred).ravel()
    return (2 * tp) / ((2 * tp) + fn + fp)


def mcc(y_true, y_pred):
    if y_pred.dtype == float:
        y_pred = threshold(y_pred)

    if np.array_equal(y_true, y_pred):
        return math.nan

    tn, fp, fn, tp = binary_confusion_mat(y_true=y_true, y_pred=y_pred).ravel()

    with np.errstate(invalid = 'raise'):
        try:
            return ((tn * tp) - (fp * fn)) / math.sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp))
        except:
            return math.nan


def auc(y_true, y_pred):

    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return None


def get_default_classification_metrics():
    return [
        ppv,
        npv,
        sensitivity,
        specificity,
        accuracy,
        balanced_accuracy,
        f1,
        mcc,
        auc
    ]


def get_default_regression_metrics():
    return [
        r2_score,
        mean_squared_error,
        mean_absolute_error
    ]


def get_classification_metrics(y_true, y_pred):
    return {
        "ppv": ppv(y_true, y_pred),
        "npv": npv(y_true, y_pred),
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "f1": f1(y_true, y_pred),
        "mcc": mcc(y_true, y_pred),
    }
