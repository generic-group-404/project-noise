from copy import deepcopy
from os import listdir

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd

import src.feature_extraction as fs
from eval_models import evaluate
from models.knn_model import KNNModel
from models.Linear_Discriminant_Analysis import LDA_model
from models.logistic_regression_model import LR_model
from models.rfc_model import RFCModel
from models.svm_model import SVMModel
from src.dataset import DataSet
from src.save_local_files import save_analysis

#TODO make a class

class ModelData:

    def __init__(self, model, n):
        """Constructor for the class"""
        super().__setattr__('__dict__', {})
        self.name = str(model)
        self.data = dict()
        self.n = n

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

    def __iadd__(self, data):
        key = list(data.keys())[0]

        if key not in self.data:
            self.data[key] = dict()
            for sub_key in data[key]:
                self.data[key][sub_key] = []

        for entry in data[key]:
            self.data[key][entry].append(data[key][entry])

        return self

    def as_dataframe(self, key):
        return pd.DataFrame(self.data[key])


def get_models_data(models, methods, n, test_n, data_path='data/'):

    interval = np.linspace(1, .1, num=n, dtype=np.float)

    container = dict()
    for model in models:
        container[str(model)] = ModelData(model, test_n)

    for method in methods:
        for inter in interval:
            print(inter)
            train_size = round(inter * 0.8, 3)
            test_size = round(inter * 0.2, 3)

            ds = DataSet(method, shuffle_data=True, test_size=test_size, train_size=train_size)
            results = evaluate(deepcopy(models), ds, n=test_n, debug=False)

            for result in results:
                container[result] += {method.__name__ : results[result]}

    return container


def create_df(data):

    for entry in data:
        for key in list(data[entry].data.keys()):
            df = data[entry].as_dataframe(key)
            save_analysis(entry, key, data[entry].n, df)


def data_exists(models, path='results/analysis_data/'):
    return True
    names = [str(x) for x in models]
    for file in listdir(path):
        if file.split('_')[0] in names:
            pass
    return True


def create_visualization(x_axis, y_axis, method_name, path='results/analysis_data/', debug=False):

    style.use('ggplot')

    min_x = []
    max_x = []

    plt.figure(figsize=(8,6))
    for file in listdir(path):
        df = pd.read_csv(path + file)
        name = file.split('_')[0]

        x = list(df[x_axis])
        y = list(df[y_axis])

        plt.plot(x, y, lw=2, label=name)

        min_x.append(min(x))
        max_x.append(max(x))

        n = file.split('_')[1]

    plt.legend(loc='best')
    
    plt.ylim(ymax=1, ymin=0)
    plt.xlim(xmax=max(max_x), xmin=min(min_x))
    
    plt.xticks(rotation=45)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    
    plt.title('{:s}_{:s}'.format(method_name, n))

    plt.savefig('figure.png')
    
    if debug:
        plt.show()


if __name__ == '__main__':

    models = [SVMModel(), SVMModel('linear'), SVMModel('poly'), LR_model(), KNNModel(), RFCModel(), LDA_model()]
    methods = [fs.mean_over_time]#, fs.mean_over_freq]
    #fs.straight_forward,
    if data_exists(models):
        create_df(get_models_data(models, methods, 100, 10))

    vis = [
        ('tr_n', 'score')
    ]
    for v in vis:
        create_visualization(v[0], v[1], 'mot', debug=True)
