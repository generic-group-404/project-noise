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

    x_train_raw = np.load('{:s}X_train.npy'.format(data_path))
    X_test_raw = np.load('{:s}X_test.npy'.format(data_path))

    X = features.straight_forward(x_train_raw)
    y = mapper.fitted

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Add/remove tested models here
    models = [SVMModel()] 

    results = dict()
    for model in models:
        model.fit(X_train, y_train)
        results[str(model)] = model.predict(X_test)

    print(results)


if __name__ == '__main__':

    evaluate()
    #add different path for data files:
    # example usage:
    # evaluate("../MyFolder/data/")
