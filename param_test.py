import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.metrics import accuracy_score

from src.dataset import DataSet
from src.feature_extraction import mean_over_time


def test_regularization(n, models, ds, visualize=False, debug=False):
    """
    Tests the SVM models and finds the best regularization strenght value C

    :param n: test size
    :param visualize: Plots the results if True. Default False.
    """
    c_range = np.linspace(0.0001, 1, num=n, endpoint=True, dtype=np.float)

    data = {}
    for model in models:
        scores = []
        for i, c in enumerate(c_range):
            model.C = c
            model.fit(ds.X_train, ds.y_train)
            y_pred = model.predict(ds.X_test)
            scores.append(accuracy_score(y_pred, ds.y_test))
            if debug:
                print(i)
        data[str(model)] = scores

    if debug:
        best = get_best(data, c_range)
        for key in best:
            print('{:s}: {}'.format(key, best[key]))

    if visualize:
        visualize_plot(data, c_range)

    return data, c_range

def test_shrinkage(n, models, ds, visualize=False, debug=False):
    """
    Tests the SVM models and finds the best regularization strenght value C

    :param n: test size
    :param visualize: Plots the results if True. Default False.
    """
    val_range = np.linspace(0.0001, 1, num=n, endpoint=True, dtype=np.float)

    data = {}
    for model in models:
        scores = []
        for i, val in enumerate(val_range):
            model.shrinkage = val
            model.fit(ds.X_train, ds.y_train)
            y_pred = model.predict(ds.X_test)
            scores.append(accuracy_score(y_pred, ds.y_test))
            if debug:
                print(i)
        data[str(model)] = scores

    if debug:
        best = get_best(data, val_range)
        for key in best:
            print('{:s}: {}'.format(key, best[key]))

    if visualize:
        visualize_plot(data, val_range)

    return data, val_range


def test_penalty(models, penalties, ds, visualize=False, test_reg=True, n=100, debug=False):

    data = {}
    names = [str(x) for x in models]
    c_range = None
    for penalty in penalties:
        for i, model in enumerate(models):
            model.penalty = penalty
            model.name = '{:s}-{:s}'.format(names[i], penalty)
            if test_reg:
                d, c_range = test_regularization(n, [model], ds)
                data.update(d)
            else:
                model.fit(ds.X_train, ds.y_train)
                y_pred = model.predict(ds.X_test)
                data[str(model)] = accuracy_score(ds.y_test, y_pred)
            if debug:
                print(i)

    if debug:
        best = get_best(data, c_range)
        for key in best:
            print('{:s}: {}'.format(key, best[key]))

    if visualize and test_reg:
        visualize_plot(data, c_range)
    elif visualize and not test_reg:
        visualize_bar(data)

    return data, c_range


def test_solver(solvers, models, ds, visualize=False, test_reg=False, test_shrink=False, n=100, debug=False):

    data = {}
    names = [str(x) for x in models]
    c_range = np.array([])
    for solver in solvers:
        for i, model in enumerate(models):
            model.solver = solver
            model.name = '{:s}-{:s}'.format(names[i], solver)
            if test_reg:
                d, c_range = test_regularization(n, [model], ds)
                data.update(d)
            elif test_shrink:
                d, c_range = test_shrinkage(n, [model], ds)
                data.update(d)
            else:
                model.fit(ds.X_train, ds.y_train)
                y_pred = model.predict(ds.X_test)
                print(model)
                data[str(model)] = accuracy_score(ds.y_test, y_pred)

        if debug:
            print(i)

    if debug:
        best = get_best(data, c_range)
        for key in best:
            print('{:s}: {}'.format(key, best[key]))

    if visualize and (test_reg or test_shrink):
        visualize_plot(data, c_range)
    elif visualize and not test_reg:
        visualize_bar(data)

    return data, c_range


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


def visualize_bar(data):

    style.use('ggplot')

    plt.bar(data.keys(), data.values(), width=.15)
    plt.show()


def visualize_plot(data, xaxis):

    style.use('ggplot')

    for key in data:
        plt.plot(xaxis, data[key], label=key)

    plt.ylim(ymax=1, ymin=0)
    plt.xlim(xmax=1, xmin=0)

    plt.yticks(rotation=90)

    plt.xlabel('c')
    plt.ylabel('acc')

    plt.title('C test size: {:d}'.format(len(xaxis)))

    plt.legend(loc='best')
    plt.show()
