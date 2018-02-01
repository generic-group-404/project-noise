# TASK 1

# Evaluate different modesl

import warnings
from time import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import src.feature_extraction as features
from models.logistic_regression_model import LR_model
from models.svm_model import SVMModel
from models.knn_model import KNNModel
from src.cross_validation import cross_validation_data
from src.label_mapper import Mapper
from src.save_local_files import save_result


def evaluate(method, models, data_path="data/", debug=True, submission=False):

    # Mapper inherits the sklearn LabelEncoder methods
    mapper = Mapper('{:s}y_train.csv'.format(data_path))

    if not submission:
        X_train, X_test, y_train, y_test = cross_validation_data(data_path)

        if debug:
            # Use only the 20% of the data set to speed up the training
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=.02, train_size=.2)

        y_test = mapper.transform(y_test)
        y_train = mapper.transform(y_train)
    else:
        X_train = np.load('{:s}X_train.npy'.format(data_path))
        X_test = np.load('{:s}X_test.npy'.format(data_path))

        y_train = mapper.fitted

        # Since there seems to be warning in the sklearn.preprocessing files we filter it.
        warnings.simplefilter("ignore")

    # Extract the features after the split to avoid information loss
    X_train = method(X_train)
    X_test = method(X_test)

    # Iterate through models
    results = dict()
    for model in models:
        meta = dict()

        # Train the model
        training_time = time()
        model.fit(X_train, y_train)
        meta['tr_time'] = time() - training_time

        # Predict the labels
        predicting_time = time()
        predictions = model.predict(X_test)
        meta['pre_time'] = time() - predicting_time

        if submission:
            result = mapper.inverse_transform(predictions)
            save_result(str(model), result)
        else:
            score = accuracy_score(y_test, predictions)
            meta['score'] = score
            if debug:
                print('{:s}: {:.6f}'.format(str(model), score))
            results[str(model)] = meta

    return results


if __name__ == '__main__':

    # Add/Remove tested models here.
    models = [SVMModel(), SVMModel('linear'), LR_model(), KNNModel()]

    # Run the evaluation function
    evaluate(features.mean_over_time, models, submission=False, debug=True)

    # add different path for data files:
    # example usage:
    # evaluate(.. data_path = "../MyFolder/data/" ..)
