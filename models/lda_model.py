from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class LDA_model(LDA):

    def __init__(self):

        LDA.__init__(
            self,
            solver='lsqr',
            shrinkage=.0041036036,
            priors=None,
            n_components=None,
            store_covariance=False,
            tol=1.0e-4
        )
        self.name = 'LDA'

    def __str__(self):
        return self.name


if __name__ == '__main__':

    import sys

    sys.path.append('.')

    from param_test import test_regularization, test_penalty, test_solver
    from src.dataset import DataSet
    from src.feature_extraction import mean_over_time
    
    ds = DataSet(method=mean_over_time)

    #solver_list = ['svd', 'lsqr', 'eigen']
    solver_list = ['lsqr', 'eigen']
    # svd and lsqr solvers were equal in terms of pure accuracy
    # best shrinkage value so far: 0.0041036036
    test_solver(solver_list, [LDA_model()], ds, visualize=True, debug=True, test_shrink=True, n=1000)
