# TASK 1

# Evaluate different modesl

import numpy as np
from sklearn.model_selection import train_test_split

import src.feature_extraction as features
from models.svm_model import SVMModel
from src.label_mapper import Mapper


#TODO convert to a class
def evaluate(data_path="data/"):

    # Mapper inherits the sklearn LabelEncoder methods
    mapper = Mapper('{:s}y_train.csv'.format(data_path))

    x_raw = np.load('{:s}X_train.npy'.format(data_path))
    y = mapper.fitted

    # Use only the 20% of the data set to speed up the training
    #X_train, X_test, y_train, y_test = train_test_split(x_raw, y, test_size=.02, train_size=.1)
    
    # FULL CASE
    X_train, X_test, y_train, y_test = train_test_split(x_raw, y, test_size=.2, train_size=.8)

    # Extract the features after the split to avoid information loss
    X_train = features.straight_forward(X_train)
    X_test = features.straight_forward(X_test)

    # Add/remove tested models here
    models = [SVMModel(), SVMModel('linear')]

    # Iterate through models
    results = dict()
    for model in models:
        model.fit(X_train, y_train)
        results[str(model)] = model.score(X_test, y_test)

    print(results)


if __name__ == '__main__':

    evaluate()
    # add different path for data files:
    # example usage:
    # evaluate("../MyFolder/data/")
