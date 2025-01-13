import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from joblib import dump
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from common.dataset import load_dataset, split_data


def train_model_with_rfecv(dataset_type="ddos", mode="auto"):
    print(
        "Running RFECV-based feature selection using XGBoost, then TabPFN for classification."
    )
    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
        X, y, return_tensor=False
    )
    xgb = XGBClassifier()
    rfecv = RFECV(
        estimator=xgb,
        step=0.1,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring="f1_weighted",
    )
    if mode == "auto":
        pfn = AutoTabPFNClassifier(max_time=1800, device="cuda")
    else:
        pfn = TabPFNClassifier(device="cuda")
    pipeline = Pipeline([("selector", rfecv), ("pfn", pfn)])

    print("Training ...")
    pipeline.fit(X_train, y_train)
    selector = pipeline.named_steps["selector"]
    print(f"Number of selected features: {selector.n_features_}")

    print("Evaluating TabPFN model on test set...")
    y_test_pred = pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="weighted")
    test_recall = recall_score(y_test, y_test_pred, average="weighted")

    print("\n=== Test Set Performance ===")
    print(f"F1-score:      {test_f1 * 100:.2f}%")
    print(f"Accuracy:      {test_accuracy * 100:.2f}%")
    print(f"Precision:     {test_precision * 100:.2f}%")
    print(f"Recall:        {test_recall * 100:.2f}%")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred, display_labels=label_encoder.classes_
    )
    plt.title("Confusion Matrix on Test Set")
    plt.show()
    dump(pipeline, "models/rfecv_tabpfn_pipeline.joblib")


if __name__ == "__main__":
    train_model_with_rfecv("ddos", mode="auto")
