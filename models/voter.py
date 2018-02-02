

class Voter:

    def __init__(self, models):

        self.__models = models

    def fit(X_train, y_train):

        for model in self.__models:

            model.fit(X_train, y_train)

    def predict(X_test, y_test):

        for model in self.__models:
            #TODO add voting
            pass

    def decision(pred_map):
        pass