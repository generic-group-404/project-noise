from sklearn.linear_model import LogisticRegression

class LR_model(LogisticRegression):

    def __init__(self):
        # Using default parameters for now
        # Class weight 
        LogisticRegression.__init__(
            self,
            penalty='l2',
            dual=False,
            tol=1e-4,
            C=1.0,
            fit_intercept=False,
            intercept_scaling=1,
            class_weight=None,
            random_state=100,
            solver='lbfgs',
            max_iter=1000,
            multi_class='multinomial',
            verbose=0,
            warm_start=False,
            n_jobs=1,
        )
        self.__name = 'LogisticRegression'

    def __str__(self):
        return self.__name
