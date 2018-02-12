import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import src.feature_extraction as feature
from src.cross_validation import cross_validation_data
from src.label_mapper import Mapper


class DataSet:

    def __init__(
        self, 
        method=None, 
        data_path='data/', 
        full=False,
        shuffle_data=False,
        norm=False,
        padd=False,
        balance=False,
        categorical=False,
        combine=None,
        **kwargs):
        """
        Constructor for the class

        :param data_path:
        :param method:
        :param full:
        :param kwargs:
        """
        super().__setattr__('__dict__', {})

        self.method = method
        if not full:
            # Loads the training and testing data using cross_validation data
            X_train, X_test, y_train, y_test = cross_validation_data(data_path)
            # Splits the data into smaller portions if needed

            if 'train_size' in kwargs:
                train_value = kwargs['train_size']

                tr_index = int(np.floor(X_train.shape[0] * train_value))

                X_train = X_train[0:tr_index, :, :]
                y_train = y_train[0:tr_index]

            if 'test_size' in kwargs:
                test_value = kwargs['test_size']

                te_index = int(np.floor(X_test.shape[0] * test_value))

                X_test = X_test[0:te_index, :, :]
                y_test = y_test[0:te_index]
        else:
            # Otherwise loads the full data set
            X_train = np.load('{:s}X_train.npy'.format(data_path))
            X_test = np.load('{:s}X_test.npy'.format(data_path))
            y_train = load_labels('{:s}y_train.csv'.format(data_path))
            y_test = y_train

        self.mapper = Mapper(y_train)

        self.y_train = self.mapper.fitted
        self.y_test = self.mapper.transform(y_test)

        if method:
            self.X_train = method(X_train)
            self.X_test = method(X_test)
        else:
            self.X_train = X_train
            self.X_test = X_test

        if norm:
            self.X_train = feature.norm_data(self.X_train)
            self.X_test = feature.norm_data(self.X_test)

        if padd:
            self.X_train = feature.pad_for_imagenet(self.X_train)
            self.X_test = feature.pad_for_imagenet(self.X_test)

        if balance:
            self.X_train, self.y_train = feature.balanced_subsample(self.X_train, self.y_train)
            self.X_test, self.y_test = feature.balanced_subsample(self.X_test, self.y_test)

        if shuffle_data:
            self.X_train, self.y_train = shuffle(self.X_train, self.y_train)

        if categorical:
            self.y_train = to_categorical(self.y_train, num_classes=15)
            self.y_test = to_categorical(self.y_test, num_classes=15)

        if combine:
            self.X_train = feature.combine_data([self.X_train for x in range(combine)])
            self.X_test = feature.combine_data([self.X_test for x in range(combine)])

        print(self.X_train.shape)
        print(self.X_test.shape)

    def __str__(self):
        return convert_method_name(self.method.__name__)

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


def convert_method_name(method_string):

    letters = [x[0] for x in method_string.split('_')]
    return ''.join(letters).upper()


def load_labels(path):
    """Reads the .csv file and returns numpy array of labels""" 
    with open(path, 'r') as f:
        labels = np.array([x.rstrip().split(",")[1] for x in f.readlines()[1:]])
    return np.array(labels)
