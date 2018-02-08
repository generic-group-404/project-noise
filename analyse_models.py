from copy import deepcopy
from os import listdir, path

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


class Analysis:

    def __init__(self, models, methods, interval_n, test_n, data_path='data/', extend=False):
        """
        Constructor for class.

        :param models:
        :param methods:
        :param interval_n:
        :param test_n:
        :param data_path:
        :param extend:
        """
        self.__models = models
        self.__methods = methods
        self.__interval_n = interval_n
        self.__test_n = test_n
        self.__path = data_path

        self.__result_path = 'results/analysis_data/'

        existing_data = self.check_data()

        if extend:
            self.create_df(self.get_models_data())
        elif existing_data:
            models_not_found = [x[0] for x in zip(models, existing_data) if x[1]]
            self.create_df(models_not_found)

    def get_models_data(self):

        interval = np.linspace(1, .1, num=self.__interval_n, dtype=np.float)

        container = dict()
        for model in self.__models:
            container[str(model)] = ModelData(model, test_n)

        for method in self.__methods:
            for inter in interval:
                train_size = round(inter * 0.8, 3)
                test_size = round(inter * 0.2, 3)

                ds = DataSet(method, shuffle_data=True, test_size=test_size, train_size=train_size)
                results = evaluate(deepcopy(models), ds, n=test_n, debug=False)

                for result in results:
                    container[result] += {method.__name__ : results[result]}

        return container

    def create_df(self, data):

        for entry in data:
            for key in list(data[entry].data.keys()):
                df = data[entry].as_dataframe(key)
                save_analysis(entry, key, data[entry].n, df, path=self.__result_path)

    def check_data(self):
        """Checks the results path for existing data and returns a boolean list of existing data for each model"""
        if path.exists(path):
            return [True for x in models]

        files = [x.split('_')[0] for x in listdir(self.__result_path)]
        out = []
        for model in self.__models:
            if str(model) in files:
                out.append(False)
            else:
                out.append(True)
        return out

    def create_visualization(self, x_axis, y_axis, method_name, path=None, debug=False, fig_path='figures'):

        style.use('ggplot')

        plt.figure(figsize=(8,6))

        min_x = []
        max_x = []

        if not path:
            data_path = self.__result_path
        else:
            data_path = path

        for file in listdir(data_path):

            if method_name in file:
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

        file_name = '{:s}_{:s}'.format(method_name, n)

        plt.title(file_name)
        plt.savefig('{:s}/{:s}.png'.format(fig_path, file_name))

        if debug:
            plt.show()


if __name__ == '__main__':

    models = [SVMModel(), SVMModel('linear'), SVMModel('poly'), LR_model(), KNNModel(), RFCModel(), LDA_model()]
    methods = [fs.mean_over_time]#, fs.mean_over_freq, fs.straight_forward]

    a = Analysis(models, methods, 10, 5)

    vis = [
        ('tr_n', 'score')
    ]
    for v in vis:
        a.create_visualization(v[0], v[1], 'mot', debug=True)
