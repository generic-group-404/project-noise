# Create the submission file

import src.feature_extraction as features
from models.knn_model import KNNModel
from models.Linear_Discriminant_Analysis import LDA_model
from models.logistic_regression_model import LR_model
from models.rfc_model import RFCModel
from models.svm_model import SVMModel
from models.voter import Voter
from src.dataset import DataSet
from src.save_local_files import save_result


def submission(model, method):
    """Trains the model with the full data set and saves the predicted labels to a submission file."""

    ds = DataSet(method, full=True)

    model.fit(ds.X_train, ds.y_train)
    y_pred = model.predict(ds.X_test)
    save_result(str(model), ds.mapper.inverse_transform(y_pred))


if __name__ == '__main__':

    models = [SVMModel(), SVMModel('linear'), SVMModel('poly'), LR_model(), KNNModel(), RFCModel(), LDA_model()]
    submission(Voter(models), features.mean_over_time)
