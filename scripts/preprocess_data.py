import os
import sys

sys.path.append('.')

import numpy as np
import pandas as pd

import src.feature_extraction as fe
from src.dataset import DataSet


def process_data(method, path='data/'):

    print('Starting the process')

    ds = DataSet(method, full=True)

    folder_path = path + str(method.__name__)

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    print(ds.X_train.shape)
    print(ds.X_test.shape)
    print(ds.y_train.shape)

    np.save(os.path.join(folder_path, 'X_train.npy'), ds.X_train)
    np.save(os.path.join(folder_path, 'X_test.npy'), ds.X_test)

    y_train = ds.mapper.inverse_transform(ds.y_train)

    with open(os.path.join(folder_path, 'y_train.csv'), 'w') as file:
        file.write('id,scene_label\n')

        for n, label in enumerate(y_train):
            file.write('{:d},{:s}\n'.format(n, label))

    print('\nFiles have been saved into: {}'.format(folder_path))

if __name__ == '__main__':

    process_data(fe.filter_by_nn)