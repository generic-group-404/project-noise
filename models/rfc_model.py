from sklearn.ensemble import RandomForestClassifier


class RFCModel(RandomForestClassifier):

    def __init__(self):

        # Toggle settings here
        RandomForestClassifier.__init__(
            self,
            n_estimators=100,
            criterion='gini',
            max_features='auto',
            max_depth=None,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0,
            max_leaf_nodes=None,
            min_impurity_decrease=0,
            bootstrap=True,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight='balanced'
        )
        self.__name = 'RandomForestClassifier'

    def __str__(self):
        """Returns the name of the model"""
        return self.__name