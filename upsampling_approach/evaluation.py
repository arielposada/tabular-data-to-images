import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from common.train_utils import evaluate_model, get_device


import torch
from common.dataset import (
    create_dataloaders,
    load_dataset,
    split_data,
)
from upsampling_approach.model import CombinedModel
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
)
from torch.utils.data import TensorDataset


def evaluate_specific_model(dataset_type="psa"):
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir / f"models/{dataset_type}"
    best_model_path = models_dir / f"best_model.pth"

    # Load the data and prepare dataloaders
    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
        X, y, return_tensor=True
    )
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    trainloader, testloader = create_dataloaders(train_dataset, test_dataset)

    device = get_device()
    input_size = X_train.shape[1]
    model = CombinedModel(input_size, num_classes).to(device)

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))

    # Evaluate the model on the test set
    y_test_true, y_test_pred = evaluate_model(model, testloader, device)
    print(f"Test accuracy: {accuracy_score(y_test_true, y_test_pred) * 100:.2f}%")
    print(
        f"Test F1-score: {f1_score(y_test_true, y_test_pred, average='weighted') * 100:.2f}%"
    )
    print(
        f"Test precision: {precision_score(y_test_true, y_test_pred, average='weighted') * 100:.2f}%"
    )
    ConfusionMatrixDisplay.from_predictions(
        y_test_true, y_test_pred, display_labels=label_encoder.classes_
    )
    plt.show()


if __name__ == "__main__":
    evaluate_specific_model("ddos")
