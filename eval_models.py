# TASK 1

# Evaluate different modesl

from copy import deepcopy
import warnings
from time import time

import numpy as np
from sklearn.metrics import accuracy_score

import src.feature_extraction as features
from models.knn_model import KNNModel
from models.Linear_Discriminant_Analysis import LDA_model
from models.logistic_regression_model import LR_model
from models.rfc_model import RFCModel
from models.svm_model import SVMModel
from src.dataset import DataSet


def evaluate(models, ds, n=1, debug=True):
    """
    Evaluate set of models. Models need to be Class with defined methods: fit(), predict() and __str__.

    :param models: list of models to be evaluated
    :param debug: Displays the loop steps for user and trains with a smaller data set.
    :param n:
    """
    # Iterate through models
    results = dict()
    for model in models:

        meta = {
            'tr_time' : [],
            'tr_n' : [],
            'pre_time' : [],
            'pre_n' : [],
            'score' : []
        }
        untrained_model = deepcopy(model)
        for _ in range(n):
            model = untrained_model
            # Train the model
            training_time = time()
            model.fit(ds.X_train, ds.y_train)
            meta['tr_time'].append(time() - training_time)
            meta['tr_n'].append(ds.X_train.shape[0])

            # Predict the labels
            predicting_time = time()
            predictions = model.predict(ds.X_test)
            meta['pre_time'].append(time() - predicting_time)
            meta['pre_n'].append(ds.X_test.shape[0])

            # Save the score
            score = accuracy_score(ds.y_test, predictions)
            meta['score'].append(accuracy_score(ds.y_test, predictions))
            if debug:
                print('{:s}: {:.6f}'.format(str(model), score))

        # Take the average over all the measurements
        mean_meta = dict()
        for key in meta:
            mean_meta[key] = np.mean(meta[key])
        
        results[str(model)] = mean_meta

    return results


if __name__ == '__main__':

    # Create the dataset
    ds = DataSet(features.mean_over_time, shuffle=True, train_size=.2, test_size=.02)

    # Add/Remove tested models here.
    models = [SVMModel(), SVMModel('linear'), SVMModel('poly'), LR_model(), KNNModel(), RFCModel(), LDA_model()]

    # Run the evaluation function
    print(evaluate(models, ds, n=10, debug=False))

    # add different path for data files:
    # example usage:
    # evaluate(.. data_path = "../MyFolder/data/" ..)
