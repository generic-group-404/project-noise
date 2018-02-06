import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import src.feature_extraction as feature
from src.cross_validation import cross_validation_data
from src.label_mapper import Mapper

class DataSet:

    def __init__(
        self, 
        method, 
        data_path='data/', 
        n=None, 
        full=False,
        shuffle_data=False,
        **kwargs):
        """
        Constructor for the class

        :param data_path:
        :param method:
        :param n:
        :param full:
        :param kwargs:
        """
        super().__setattr__('__dict__', {})

        if not full:
            # Loads the training and testing data using cross_validation data
            X_train, X_test, y_train, y_test = cross_validation_data(data_path)
            # Splits the data into smaller portions if needed
            if shuffle_data:
                X_train, y_train = shuffle(X_train, y_train)

            if 'train_size' in kwargs and 'test_size' in kwargs:
                test_value = kwargs['test_size']
                train_value = kwargs['train_size']

                tr_index = int(np.floor(X_train.shape[0] * train_value))
                te_index = int(np.floor(X_test.shape[0] * test_value))

                X_train = X_train[0:tr_index, :, :]
                y_train = y_train[0:tr_index]
        else:
            # Otherwise loads the full data set
            X_train = np.load('{:s}X_train.npy'.format(data_path))
            X_test = np.load('{:s}X_test.npy'.format(data_path))
            y_train = load_labels('{:s}y_train.csv'.format(data_path))
            y_test = y_train

        if n:
            self.X_train = feature.split_extract_data(method, n, 0, X_train)
            self.X_test = feature.split_extract_data(method, n, 0, X_test)
        else:
            self.X_train = method(X_train)
            self.X_test = method(X_test)

        self.mapper = Mapper(y_train)

        self.y_train = self.mapper.fitted
        self.y_test = self.mapper.transform(y_test)

    def __getattr__(self, key):
        """
        Gets class attribute
        Raises AttributeError if key is invalid
        """
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError

    def __setattr__(self, key, value):
        """
        Sets class attribute according to value
        If key was not found, new attribute is added
        """
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            super().__setattr__(key, value)


def load_labels(path):
    """Reads the .csv file and returns numpy array of labels""" 
    with open(path, 'r') as f:
        labels = np.array([x.rstrip().split(",")[1] for x in f.readlines()[1:]])
    return np.array(labels)
