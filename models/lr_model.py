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
            C=0.908918018018018,
            fit_intercept=False,
            intercept_scaling=1,
            class_weight=None,
            random_state=100,
            solver='saga',
            max_iter=2000,
            multi_class='multinomial',
            verbose=0,
            warm_start=False,
            n_jobs=1,
        )
        self.name = 'LR'

    def __str__(self):
        return self.name

if __name__ == '__main__':

    import sys

    sys.path.append('.')

    from param_test import test_regularization, test_penalty
    from src.dataset import DataSet
    from src.feature_extraction import mean_over_time
    
    ds = DataSet(method=mean_over_time)
    #test_regularization(n=1000, models=[LR_model()], ds=ds, visualize=True)

    penalties_list = ['l1', 'l2']
    #test_penalty([LR_model()], penalties_list, ds, visualize=True, n=1000, debug=True)
    test_regularization(1000, [LR_model()], ds, visualize=True, debug=True)
    #best so far: 0.908918018018018
