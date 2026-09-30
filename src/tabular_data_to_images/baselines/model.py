from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from tabular_data_to_images.lstm.model import TabularLSTMClassifier
from tabular_data_to_images.transformer.model import TabularTransformerClassifier
from tabular_data_to_images.baselines.image_models import (
    DeepInsightClassifier,
    TACClassifier,
    IGTDClassifier,
    RefinedClassifier,
)


def get_model(model_type, num_classes, dataset_type=None, seed=42):
    """
    Initialize and return a machine learning model based on the specified type.

    Args:
    model_type (str): One of "xgboost", "random_forest", "svm",
        "logistic_regression", "lstm", "transformer", "deepinsight", "tac",
        "igtd", "refined".
    num_classes (int): The number of unique classes in the target dataset.
    dataset_type (str): Which dataset this model will be trained on. Only
        needed by "refined", which loads a precomputed feature->pixel
        mapping specific to the dataset.
    seed (int): Random seed, forwarded to sklearn/xgboost estimators and to
        the CNN-based image transforms (torch's own RNG state should be
        seeded separately, e.g. via `common.train_utils.set_seed`).

    Returns:
    model: An instance of a machine learning model exposing
        `.fit(X, y)` / `.predict(X)`.
    """
    image_num_epochs = 10 if dataset_type == "covid" else 200

    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
            num_class=num_classes if num_classes > 2 else None,
            random_state=seed,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed)
    elif model_type == "svm":
        return SVC(random_state=seed)
    elif model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    elif model_type == "lstm":
        return TabularLSTMClassifier()
    elif model_type == "transformer":
        return TabularTransformerClassifier()
    elif model_type == "deepinsight":
        return DeepInsightClassifier(num_epochs=image_num_epochs)
    elif model_type == "tac":
        return TACClassifier(num_epochs=image_num_epochs)
    elif model_type == "igtd":
        return IGTDClassifier(num_epochs=image_num_epochs, seed=seed)
    elif model_type == "refined":
        return RefinedClassifier(dataset_type=dataset_type, num_epochs=image_num_epochs)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
