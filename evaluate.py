from sklearn.metrics import f1_score

def binarize(pred, threshold=0.5):
    return pred > threshold

def mean_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='samples')
