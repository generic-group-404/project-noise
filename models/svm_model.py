from sklearn.svm import LinearSVC


class SVMModel(LinearSVC):

    def __init__(self):

        # Toggle settings here
        LinearSVC.__init__(
            self
            #TODO add settings
        )
        self.__name = 'SVM Model'

    def __str__(self):
        """Returns the name of the model"""
        return self.__name
