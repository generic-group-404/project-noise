from collections import Counter

import numpy as np
from librosa import db_to_power, load
from librosa.decompose import nn_filter
from librosa.feature import mfcc, rmse
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle


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


def norm_data(raw):
    """Normalizes every sample in the data"""
    return np.array([normalize(x) for x in raw])


def image_format(raw):
    """Formats the data to simple grayscale format"""
    return np.array([x.reshape(x.shape[0], x.shape[1], 1) for x in raw])


def combine_data(*data):
    """Stacks the data"""
    out = []
    for i in range(data[0].shape[0]):
        out.append(np.dstack(tuple([x[i] for x in data])))
    return np.array(out)


def pad_for_imagenet(raw):
    """Adds the extra zeros in the end of raw data"""
    data = []
    for row in raw:
        empty = np.zeros((48, row.shape[1]))
        empty[0:row.shape[0],:] = row
        data.append(empty)
    return np.array(data)


def mfcc_spec(raw):
    """Calculates the Mel-frequency cepstral coefficients for each sample in data"""
    return np.array([mfcc(S=x, n_mfcc=x.shape[0]) for x in raw])


def rmse_spec(raw):
    """Calculates the root-mean-square energy for each sample in data."""
    return np.array([rmse(S=x) for x in raw])


def filter_by_nn(raw):
    """Filters the data using nn -method"""
    return np.array([nn_filter(x, aggregate=np.median, metric='cosine') for x in raw])


def balanced_subsample(X, y):
    """Balances the data to even sample sizes"""
    count = Counter(y)
    smallest = sorted(count.values())[0]
    data = [(value, X[(y == value)]) for value in count.keys()]

    xs = []
    ys = []
    for ci, this_xs in data:
        if len(this_xs) > smallest:
            np.random.shuffle(this_xs)

        x_ = this_xs[:smallest]
        y_ = np.empty(smallest, dtype=int)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs, ys = shuffle(np.concatenate(xs), np.concatenate(ys))

    return xs, ys
