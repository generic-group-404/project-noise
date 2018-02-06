from collections import Counter

import numpy as np

class Voter:

    def __init__(self, models, n=3):
        """
        Constructor for Voter class

        :param models:
        :param n:
        """
        self.__models = models
        self.__name = 'voter'
        self.__n = n

    def __str__(self):
        return self.__name

    def fit(self, X_train, y_train):
        """Trains the modesl with given data X_train and labels y_train"""
        for model in self.__models:
            model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict best labels for given samples X_test"""
        return np.array([self.vote(np.reshape(x,(1, x.shape[0]))) for x in X_test])

    def vote(self, sample):
        """Predicts the sample propability for each model and let them vote for the best label"""
        zeros = [0 for x in range(self.__n)]

        data = {'proba' : zeros, 'class' : zeros}
        for model in self.__models:
            pred_list = list(model.predict_proba(sample)[0])
            pred_value = max(pred_list)
            worse = min(data['proba'])
            if pred_value > worse:
                index = data['proba'].index(worse)
                data['proba'][index] = pred_value
                data['class'][index] = pred_list.index(pred_value)

        count = Counter(data['class'])

        if len(list(count.values())) > 2:
            # Listen to the most certain classifier in the case of draw
            return data['class'][data['proba'].index(max(data['proba']))]
        else:
            # Else we pick the highest voted class
            return sorted(count, key=lambda x: count[x], reverse=True)[0]

    #TODO add earn the voting threshold
    #TODO add different council sizes