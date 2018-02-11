import numpy as np


def cross_validation_data(data_path='data/', cross_validation_path='data/'):
    """Splits the data according to the crossvalidation file"""

    X_raw = np.load('{:s}X_train.npy'.format(data_path))

    X = {'train':[], 'test':[]}
    y = {'train':[], 'test':[]}
    with open('{:s}crossvalidation_train.csv'.format(cross_validation_path), 'r') as f:

        for row in zip(f.readlines()[1:], X_raw):
            parts = row[0].rstrip().split(",")
            X[parts[2]].append(row[1])
            y[parts[2]].append(parts[1])

    return np.array(X['train']), np.array(X['test']), np.array(y['train']), np.array(y['test'])
