from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class LDA_model(LDA):

    def __init__(self):

        LDA.__init__(
            self,
            solver='svd',
            shrinkage=None,
            priors=None,
            n_components=None,
            store_covariance=False,
            tol=1.0e-4
        )
        self.__name = 'LDA'

    def __str__(self):
        return self.__name
