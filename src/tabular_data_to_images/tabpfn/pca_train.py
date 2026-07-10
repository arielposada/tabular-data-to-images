from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from tabular_data_to_images.common.dataset import load_dataset, split_data
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model_cv_test(dataset_type="psa", n_splits=5, mode="auto"):
    print(
        f"Evaluating TabPFN model on {dataset_type} dataset with {n_splits}-fold CV..."
    )

    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
        X, y, return_tensor=False
    )

    if mode == "auto":
        clf = AutoTabPFNClassifier(max_time=120)  # 120 seconds tuning time
    else:
        clf = TabPFNClassifier()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=22)),
            ("tabpfn", clf),
        ]
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = ["f1_weighted", "accuracy", "precision_weighted", "recall_weighted"]
    cv_results = cross_validate(
        pipeline, X_train, y_train, scoring=scoring, cv=skf, n_jobs=-1
    )

    print(f"\n===== {n_splits}-fold CV Results on Training Set =====")
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        print(f"{metric:<22}: {scores.mean():.4f} ± {scores.std():.4f}")

    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="weighted")
    test_recall = recall_score(y_test, y_test_pred, average="weighted")

    print(f"\n===== Final Evaluation on Hold-Out Test Set =====")
    print(f"Test F1-score:       {test_f1:.4f}")
    print(f"Test Accuracy:       {test_accuracy:.4f}")
    print(f"Test Precision:      {test_precision:.4f}")
    print(f"Test Recall:         {test_recall:.4f}")
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
    plt.title("Confusion Matrix on Hold-Out Test Set")
    plt.show()


if __name__ == "__main__":
    train_model_cv_test("ddos", n_splits=5)
