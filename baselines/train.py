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
from common.dataset import load_dataset, split_data
from model import get_model


def train_model(dataset_type="psa", model_type="xgboost"):
    print(f"Training {model_type} model on {dataset_type} dataset...")

    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
        X, y, return_tensor=False
    )

    # Use the auxiliary function to get the model
    model = get_model(model_type, num_classes)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_test_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average="weighted")
    test_recall = recall_score(y_test, y_test_pred, average="weighted")

    print(f"Test F1-score: {test_f1 * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Precision: {test_precision * 100:.2f}%")
    print(f"Test Recall: {test_recall * 100:.2f}%")

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred, display_labels=label_encoder.classes_
    )
    plt.show()


if __name__ == "__main__":
    # train_model("ddos", "xgboost")
    train_model("ddos", "random_forest")
