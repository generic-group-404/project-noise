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
