from keras import backend

def fbeta(y_true, y_pred, beta=2):

    """
    compute fbeta score for multi-class/label classification
    """

    # clip predictions
    y_pred = backend.clip(y_pred, 0, 1)

    # calculate elements
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)

    # calculate precision
    p = tp / (tp + fp + backend.epsilon())

    # calculate recall
    r = tp / (tp + fn + backend.epsilon())

    # calculate fbeta, averaged across each class
    bb = beta ** 2
    return backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
