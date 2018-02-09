# Create the submission file

import src.feature_extraction as features
from models.knn_model import KNN_model
from models.lda_model import LDA_model
from models.lr_model import LR_model
from models.rfc_model import RFC_model
from models.svm_model import SVM_model
from models.voter import SimpleVoter
from src.dataset import DataSet
from src.save_local_files import save_submission


def submission(model, method):
    """Trains the model with the full data set and saves the predicted labels to a submission file."""

    ds = DataSet(method, full=True)

    model.fit(ds.X_train, ds.y_train)
    y_pred = model.predict(ds.X_test)
    save_submission(str(model), ds.mapper.inverse_transform(y_pred))


if __name__ == '__main__':

    #models = [SVM_model(), SVM_model('linear'), SVM_model('poly'), LR_model(), KNN_model(), RFC_model(), LDA_model()]
    best_models = [SVM_model('linear'), SVM_model('poly'), LR_model(), LDA_model()]
    submission(SimpleVoter(best_models, council=2), features.mean_over_time)
