import os
import sys
import time
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent)
)  # for igtd_functions / igtd_transformer

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import TensorDataset

from tabular_data_to_images.common.dataset import (
    load_dataset,
    split_data,
    create_dataloaders,
)
from tabular_data_to_images.common.model import SimpleCNN
from tabular_data_to_images.common.train_utils import (
    get_criterion,
    get_device,
    get_optimizer,
    train_epoch,
    evaluate_model,
)

from tabular_data_to_images.IGTD.igtd_transformer import IGTDTransformer


def save_results(base_name, results, n_features, conf_matrix_plt, timing):
    models_dir = Path(__file__).resolve().parent / "models" / base_name
    models_dir.mkdir(exist_ok=True, parents=True)

    with open(models_dir / f"{base_name}_results.txt", "w") as f:
        res = f"Results for {base_name} model\n"
        res += str(results)
        res += "\n"
        res += f"Number of selected features: {n_features}"
        res += "\n results: "
        values = ",".join([str(x) for x in results.values()])
        res += f"\n{values}"
        res += f"\ntiming: {timing}"
        f.write(str(res))

    if conf_matrix_plt is not None:
        conf_matrix_plt.savefig(models_dir / f"{base_name}_confusion_matrix.png")

    return models_dir


def train(dataset_type="ddos", seed=42, max_step=1000, image_size=36):
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir / "models" / dataset_type
    models_dir.mkdir(exist_ok=True, parents=True)
    best_model_path = models_dir / "best_model.pth"
    mapping_path = models_dir / "igtd_mapping.pkl"

    lr = 1e-3
    batch_size = 32
    num_epochs = 200
    print_epochs = 10
    num_classes = 2

    device = get_device()

    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
        X, y, return_tensor=False, random_state=seed
    )

    # --- Fit the IGTD feature<->pixel mapping on the training set only ---
    t0 = time.perf_counter()
    transformer = IGTDTransformer(
        num_row=image_size,
        num_col=image_size,
        max_step=max_step,
        random_state=seed,
        save_folder=str(models_dir / "igtd_fit_diagnostics"),
    )
    transformer.fit(X_train)
    fit_time = time.perf_counter() - t0
    transformer.save(mapping_path)

    t0 = time.perf_counter()
    X_train_img = transformer.transform(X_train)
    X_test_img = transformer.transform(X_test)
    transform_time = time.perf_counter() - t0

    X_train_tensor = torch.tensor(X_train_img, dtype=torch.float32).unsqueeze(
        1
    )  # (N, 1, H, W)
    X_test_tensor = torch.tensor(X_test_img, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    trainloader, testloader = create_dataloaders(
        train_dataset, test_dataset, batch_size
    )

    model = SimpleCNN(in_channels=1, num_classes=num_classes, image_size=image_size).to(
        device
    )
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=lr)

    best_f1_score = 0.0
    train_start = time.perf_counter()
    for epoch in range(num_epochs):
        loss, accuracy = train_epoch(model, trainloader, optimizer, criterion, device)
        if epoch % print_epochs == 0 or epoch == num_epochs - 1:
            y_test_true, y_test_pred = evaluate_model(model, testloader, device)
            val_f1 = f1_score(y_test_true, y_test_pred, average="weighted") * 100
            if val_f1 > best_f1_score:
                best_f1_score = val_f1
                torch.save(model.state_dict(), best_model_path)
            print(f"[{epoch}] loss: {loss:.3f}, val F1-score: {val_f1:.2f}%")
    train_time = time.perf_counter() - train_start

    model.load_state_dict(torch.load(best_model_path))
    y_test_true, y_test_pred = evaluate_model(model, testloader, device)
    test_f1 = f1_score(y_test_true, y_test_pred, average="weighted")
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    test_precision = precision_score(y_test_true, y_test_pred, average="weighted")
    test_recall = recall_score(y_test_true, y_test_pred, average="weighted")

    print(f"Test F1-score: {test_f1 * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Precision: {test_precision * 100:.2f}%")
    print(f"Test Recall: {test_recall * 100:.2f}%")

    ConfusionMatrixDisplay.from_predictions(
        y_test_true, y_test_pred, display_labels=label_encoder.classes_
    )
    base_name = f"igtd_{dataset_type}"
    timing = {
        "igtd_fit_seconds": fit_time,
        "igtd_transform_seconds": transform_time,
        "cnn_train_seconds": train_time,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    save_results(
        base_name,
        {
            "f1_score": round(test_f1, 4),
            "accuracy": round(test_accuracy, 4),
            "precision": round(test_precision, 4),
            "recall": round(test_recall, 4),
        },
        X_train.shape[-1],
        plt,
        timing,
    )
    print("Timing:", timing)


if __name__ == "__main__":
    train("ddos")
