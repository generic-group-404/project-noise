from sklearn.neighbors import KNeighborsClassifier


class KNNModel(KNeighborsClassifier):

    def __init__(self):

        KNeighborsClassifier.__init__(
            self,
            weights='uniform',
            algorithm='auto',
            leaf_size='30',
            p=2,
            metric='minkowski',
            metric_params=None,
            n_jobs='1',
        )
        self.__name = 'k-NN'

    def __str__(self):
        return self.__name
