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
from tabular_data_to_images.common.dataset import load_dataset, split_data
from tabular_data_to_images.common.train_utils import get_feature_selector
from tabular_data_to_images.paths import MODELS_DIR


def train_model_with_rfecv(dataset_type="ddos", mode="auto"):
    print(
        "Running RFECV-based feature selection using XGBoost, then TabPFN for classification."
    )
    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
        X, y, return_tensor=False
    )
    xgb = XGBClassifier()
    rfecv = get_feature_selector(xgb, name="rfecv")
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
    out_dir = MODELS_DIR / "tabpfn_rfecv"
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(pipeline, out_dir / "rfecv_tabpfn_pipeline.joblib")


if __name__ == "__main__":
    train_model_with_rfecv("ddos", mode="auto")
