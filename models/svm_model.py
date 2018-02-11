from sklearn.svm import SVC


class SVM_model(SVC):

    def __init__(self, kernel='rbf'):

        # Toggle settings here
        # Best C values for each kernels so far:
        c_map = {'rbf':1, 'linear':0.93939999, 'poly':.012118081}
        #c_map = {'rbf':1, 'linear':1, 'poly':1}
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

    from param_test import ParamTester

    from src.dataset import DataSet
    import src.feature_extraction as fe

    ds = DataSet(method=fe.mfcc_spec, shuffle=True)

    test = ParamTester(ds, SVM_model('poly'), iter_method = {'C':100}, debug=True)
    test.run()
    test.save_results()
    test.plot()
