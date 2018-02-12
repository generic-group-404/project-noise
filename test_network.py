import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from matplotlib import style
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC

import src.feature_extraction as features
from network.rnn1 import RNN1
from network.vgg16_base import VGG16_net
from src.dataset import DataSet
from src.save_local_files import get_fig_file, save_submission
from submission import submission


def plot_training(history, net, epochs, debug=False):

    style.use('ggplot')

    plots = {
        'accuracy':
            ['acc','val_acc']
        ,
        'loss':
            ['loss','val_loss']
    }
    for i, key in enumerate(plots.keys()):
        plt.figure(i)
        print(key)
        for subkey in plots[key]:
            plt.plot(history.history[subkey], label=subkey)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(loc='best')

        plt.savefig(get_fig_file(str(net), key, '{}e'.format(epochs)))

        if debug:
            plt.show()


def train(X_train, y_train, val_X, val_y, net, epochs=50):

    history = net.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1, validation_data=(val_X, val_y))

    y_train = np.argmax(y_train, axis=1)
    clf = LinearSVC()
    x_f = net.get_features(X_train)
    print(x_f)
    clf.fit(x_f, y_train)
    y = clf.predict(net.get_features(val_X))
    val_y = np.argmax(val_y, axis=1)
    print(accuracy_score(val_y, y))
    return history, net, clf


def submmiss_clf(net, clf, ds):

    x_features = net.get_features(ds.X_test)

    y_pred = clf.predict(x_features)

    save_submission('{}-clf'.format(str(net)), ds.mapper.inverse_transform(y_pred))


def predict(X_test, y_test, net, threshold=.7):
    
    pred = np.argmax(net.predict(X_test), axis=1)
    y_test = np.argmax(y_test, axis=1)
    score = accuracy_score(y_test, pred)
    print('Predicting test samples: {}'.format(score))

    # Do something with the knowledge
    return score > threshold


def test_rnn():
    
    ds = DataSet(features.mfcc_spec, data_path='data/', shuffle=True, balance=True, categorical=True)
    
    r = RNN1()

    epochs_count = 1

    history, net, clf = train(ds.X_train, ds.y_train, ds.X_test, ds.y_test, r, epochs=epochs_count)

    plot_training(history, net, epochs_count)

    if predict(ds.X_test, ds.y_test, net):
        ds_sub = DataSet(full=True)
        submission(net, None, ds_sub, network=True, trained=True)


def test_vgg16():
    
    ds = DataSet(data_path='data/', shuffle=True, balance=True, categorical=True, padd=True, combine=3)
    
    r = VGG16_net()

    epochs_count = 10

    history, net, clf = train(ds.X_train, ds.y_train, ds.X_test, ds.y_test, r, epochs=epochs_count)

    #plot_training(history, net, epochs_count)

    if predict(ds.X_test, ds.y_test, net):
        full_ds = DataSet(data_path='data/', padd=True, combine=3)

        #submission(net, None, full_ds, network=True, trained=True)
        submmiss_clf(net, clf, full_ds)

if __name__ == '__main__':

    test_vgg16()
    #test_rnn()
