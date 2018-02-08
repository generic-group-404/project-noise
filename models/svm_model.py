from sklearn.svm import SVC


class SVM_model(SVC):

    def __init__(self, kernel='rbf'):

        # Toggle settings here
        # Best C values for each kernels so far:
        c_map = {'rbf':1, 'linear':0.93939999, 'poly':.012118081}
        try:
            SVC.__init__(
                self,
                C=c_map[kernel],
                kernel=kernel,
                gamma='auto',
                coef0=0.0,
                probability=True,
                shrinking=True,
                tol=1e-3,
                cache_size=200,
                class_weight=None,
                verbose=False,
                max_iter=-1,
                decision_function_shape='ovr',
                random_state=None,
            )
        except KeyError:
            print('False kernel')
        self.name = 'SVM-{:s}'.format(kernel)

    def __str__(self):
        """Returns the name of the model"""
        return self.name

if __name__ == '__main__':

    import sys

    sys.path.append('.')

    from param_test import test_regularization

    from src.dataset import DataSet
    from src.feature_extraction import mean_over_time
    
    ds = DataSet(method=mean_over_time, shuffle=True, n=2, ravel_test=True, ax=1)

    models = [SVM_model('rbf'), SVM_model('linear'), SVM_model('poly')]
    #models = [SVM_model('rbf')]
    test_regularization(n=100, models=models, ds=ds, visualize=True, debug=True)
