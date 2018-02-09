import numpy as np


def straight_forward(raw):
    """Converts matrixes to vector features"""
    return np.array([x.ravel() for x in raw])


def mean_over_time(raw):
    """Takes a mean value for each data matrix"""
    return np.array([np.mean(x, axis=1) for x in raw])


def mean_over_freq(raw):
    """Takes a mean value for each data matrix"""
    return np.array([np.mean(x, axis=0) for x in raw])


def split_extract_data(method, n, axis, data):
    """Splits and extracts features from the data according to the n"""
    return np.array([method(np.split(d, n, axis=axis)) for d in data])


def ravel_splitted_data(X, y):
    """Ravels the splitted data"""
    r_X = []
    r_y = []
    for i, row in enumerate(X):
        for entry in row:
            r_X.append(entry)
            r_y.append(y[i])
    return np.array(r_X), np.array(r_y)
