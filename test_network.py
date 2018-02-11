import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from matplotlib import style
from sklearn.metrics import accuracy_score

import src.feature_extraction as features
from network.rnn1 import RNN1
from network.vgg16_base import VGG16_net
from src.dataset import DataSet
from src.save_local_files import get_fig_file

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
        
        if debug:
            plt.show()

        plt.savefig(get_fig_file(str(net), key, '{}e'.format(epochs_count)))


def train(X_train, y_train, val_X, val_y, net, epochs=50):

    history = net.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1, validation_data=(val_X, val_y))

    return history, net


def predict(X_test, y_test, net, threshold=.7):
    
    pred = np.argmax(net.predict(X_test), axis=1)
    score = accuracy_score(ds.y_test, pred)
    print(score)
    if score > threshold:
        pass

if __name__ == '__main__':

    # Create the dataset
    ds = DataSet(shuffle=True, padd=True)
    #X_train = ds.X_train
    #X_test = ds.X_test

    # Convert data to (sample_count, x, y, z)
    X_train = features.combine_data(ds.X_train, ds.X_train, ds.X_train)
    X_test = features.combine_data(ds.X_test, ds.X_test, ds.X_test)

    X_train, y_train = features.balanced_subsample(X_train, ds.y_train)
    X_test, y_test = features.balanced_subsample(X_test, ds.y_test)

    y_train = to_categorical(y_train, num_classes=15)
    y_test = to_categorical(y_test, num_classes=15)

    print(X_train.shape)
    print(X_test.shape)

    #r = RNN1()
    r = VGG16_net()

    epochs_count = 75

    history, net = train(X_train, y_train, X_test, y_test, r, epochs=epochs_count)

    plot_training(history, net, epochs_count, debug=True)

    predict(X_test, ds.y_test, net)