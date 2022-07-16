import math

from sklearn.metrics import confusion_matrix


def confusion_mat(y_true, y_pred):
    return confusion_matrix(y_true=y_true, y_pred=y_pred)


def ppv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return tp / (tp + fp)


def npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return tn / (tn + fn)


def sensitivity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return tp / (tp + fn)


def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return tn / (tn + fp)


def accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return (tn + tp) / (tp + tn + fp + fn)


def balanced_accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return ((tn / (tn + fp)) + (tp / (tp + fn))) / 2


def f1(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return (2 * tp) / ((2 * tp) + fn + fp)


def mcc(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    return ((tn * tp) - (fp * fn)) / math.sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp))


def auc(y_true, y_pred):
    raise NotImplementedError
    # TODO implement this


def get_all(y_true, y_pred):
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
