import numpy as np


def straight_forward(raw):
    """Converts matrixes to vector features"""
    return np.array([x.ravel() for x in raw])

#TODO add more feature extraction