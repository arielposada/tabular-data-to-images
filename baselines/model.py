from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


def get_model(model_type, num_classes):
    """
    Initialize and return a machine learning model based on the specified type.

    Args:
    model_type (str): The type of model to initialize ('xgboost' or 'random_forest').
    num_classes (int): The number of unique classes in the target dataset.

    Returns:
    model: An instance of a machine learning model.
    """
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
            num_class=num_classes if num_classes > 2 else None,
            random_state=42,
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
