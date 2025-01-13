import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score

import optuna

from common.dataset import load_dataset, split_data
from tabpfn import TabPFNClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def train_model(dataset_type="psa"):
    print(f"Training TabPFN model on {dataset_type} dataset...")

    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
        X, y, return_tensor=False
    )

    def objective(trial):
        n_components = trial.suggest_int("n_components", 5, min(200, X_train.shape[1]))
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_components)),
                ("tabpfn", TabPFNClassifier()),
            ]
        )
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            scoring="f1_weighted",
            cv=5,
        )
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.enqueue_trial({"n_components": 50})
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    print("=== Optimization Results ===")
    print(f"Best Score: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")

    # Refit a pipeline using the best found hyperparams
    best_n_components = study.best_params["n_components"]

    best_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=best_n_components)),
            ("tabpfn", TabPFNClassifier()),
        ]
    )
    best_pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_test_pred = best_pipeline.predict(X_test)

    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="weighted")
    test_recall = recall_score(y_test, y_test_pred, average="weighted")

    print("\n=== Test Set Performance ===")
    print(f"F1-score:       {test_f1 * 100:.2f}%")
    print(f"Accuracy:       {test_accuracy * 100:.2f}%")
    print(f"Precision:      {test_precision * 100:.2f}%")
    print(f"Recall:         {test_recall * 100:.2f}%")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred, display_labels=label_encoder.classes_
    )
    plt.show()


if __name__ == "__main__":
    train_model("ddos")
