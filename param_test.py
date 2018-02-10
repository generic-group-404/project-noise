import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import accuracy_score
import pandas as pd
from src.dataset import DataSet
from src.feature_extraction import mean_over_time
from src.save_local_files import get_fig_file, save_analysis

class ParamTester:

    def __init__(
        self, 
        dataset,
        *models,
        iter_method = [],
        static_method = [],
        cross_test=False, 
        visualize=False, 
        debug=False
        ):
        """
        Constructor for the class.

        :param dataset: DataSet object containing the training and testing data
        :param cross_test: Mode to test methods against each other, otherwise runs the each
                        method as separate case.
        :param visualize: Mode to visualize the results 
        :param debug: Mode to debug, prints steps to cmd
        :param n: size of the iteration
        :param models: models to be tested
        :param methods: methods and method params to be tested
        """
        super().__setattr__('__dict__', {})

        self.iter = iter_method
        self.static = static_method
        self.debug = debug
        self.ds = dataset
        self.cross_test = cross_test
        self.models = models

        self.data = dict()
        self.ranges = dict()

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

    def run(self):
        """
        """
        self.plots = []
        self.hists = []
        for model in self.models:
            if self.cross_test:
                for static in self.static:
                    modified_model, _ = self.test_framework(model, static)
                    for method in self.iter:
                        _, results = self.test_framework(modified_model, method)
                        self.data[str(model)] = results
                        self.plots.append(str(model))
            else:
                if self.iter:
                    for method in self.iter:
                        _, results = self.test_framework(model, method)
                        self.data[str(model)] = results
                        self.plots.append(str(model))

                if self.static:
                    for method in self.static:
                        _, results = self.test_framework(model, method)
                        self.data[str(model)] = results
                        self.hists.append(str(model))

        print(self.data)

    def save_results(self, path='results'):

        if self.iter:
            for key in self.plots:
                data = dict()
                data['score'] = self.data[key]
                data['n'] = self.ranges[key]
                df = pd.DataFrame(data)
                save_analysis(key, df, 'n{:d}'.format(len(data['n'])), folder='/param_tests', path=path)

        #TODO
        if self.hists:
            for method in self.static:
                df = pd.DataFrame(self.hists)

    def test_framework(self, model, method):

        scores = []
        if method in self.iter:
            start = 0.0001
            end = 1.0
            test_range = np.linspace(start, end, num=self.iter[method], endpoint=True)
            self.ranges[str(model)] = test_range
            for i, val in enumerate(test_range):
                model.__dict__[method] = val
                scores.append(self.get_score(model))
                if self.debug:
                    print('Iteration: {:d}'.format(i))
        else:
            model.__dict__[method] = self.kwargs[method]
            scores.append(self.get_score(model))
            name = str(model).split('-')[0]
            model.name = '{:s}-{:s}'.format(name, self.static[method])
        return model, scores

    def get_score(self, model):

        model.fit(self.ds.X_train, self.ds.y_train)
        pred = model.predict(self.ds.X_test)
        return accuracy_score(pred, self.ds.y_test)

    def plot(self, name='test', **kwargs):
        
        style.use('ggplot')

        for key in self.plots:    
            plt.plot(self.ranges[key], self.data[key], label=key)

            plt.ylim(ymax=1, ymin=0)
            plt.xlim(xmax=1, xmin=0)

            if 'xlabel' in kwargs:
                plt.xlabel(kwargs['xlabel'])
            if 'ylabel' in kwargs:
                plt.ylabel(kwargs['ylabel'])

            plt.title('Test size: {:d}'.format(len(self.data[key])))

            plt.legend(loc='best')
            plt.savefig(get_fig_file(name, str(self.ds), 'plot'))

            if self.debug:
                plt.show()

        for key in self.hists:
            plt.bar(self.data.keys(), self.data.values(), width=.15)
            plt.savefig(get_fig_file(name, str(self.ds), 'bar'))
            if self.debug:
                plt.show()


def get_best(data, c):
    """Iterate through the results and selects the best result"""
    best = {'score':0, 'c':None, 'model':None}
    for key in data:
        if c.any():
            acc = max(data[key])
            if(acc > best['score']):
                best['score'] = acc
                best['c'] = c[data[key].index(acc)]
                best['model'] = key
        else:
            if data[key] > best['score']:
                best['score'] = data[key]
                best['model'] = key
    return best
