from sklearn.svm import SVC


class SVMModel(SVC):

    def __init__(self, kernel='rbf'):

        # Toggle settings here
        SVC.__init__(
            self,
            C=1.0,
            kernel=kernel,
            gamma='auto',
            coef0=0.0,
            probability=False,
            shrinking=True,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape='ovr',
            random_state=None,
        )
        self.__name = 'SVM-{:s}'.format(kernel)

    def __str__(self):
        """Returns the name of the model"""
        return self.__name
