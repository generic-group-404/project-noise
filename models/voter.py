from collections import Counter

import numpy as np

class Voter:

    def __init__(self, models):
        """Constructor for the Voter class"""
        self.__models = models
        self.__name = 'voter'

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

        data = {'proba' : [0, 0, 0], 'class' : [0, 0, 0], 'models' : []}
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

