import numpy as np

def mean_square_error(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mean_square_error_derivative(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size