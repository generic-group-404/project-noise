from collections import Counter

import numpy as np

class SimpleVoter:

    def __init__(self, models, council=3):
        """
        Constructor for Voter class

        :param models: Models that will be voting
        :param council: Council size, how many models get to vote
        """
        self.__models = models
        self.__council = council

        self.__name = 'SimpleVoter'

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
        zeros = [0 for x in range(self.__council)]

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

        if len(list(count.values())) > int(np.ceil(self.__council / 2)):
            # Listen to the most certain classifier in the case of draw
            return data['class'][data['proba'].index(max(data['proba']))]
        else:
            # Else we pick the highest voted class
            return sorted(count, key=lambda x: count[x], reverse=True)[0]


class PartialVoter(SimpleVoter):

    def __init__(self, models, council=3, threshold=None):
        """
        Constructor for Voter class

        :param models:
        :param council:
        :param threshold:
        """
        SimpleVoter.__init__(self, models, council)

        self.__threshold = threshold

        self.__name = 'PartialVoter'

    def __str__(self):
        return self.__name

    def predict(self, X_test):
        """Predict best labels for given samples X_test"""
        return np.array([self.partial_vote(x) for x in X_test])

    def partial_vote(self, data_slice):
        
        result = [self.vote(np.reshape(x,(1, x.shape[0]))) for x in data_slice]
        count = Counter(result)
        print(count)
        return sorted(count, key=lambda x: count[x], reverse=True)[0]
