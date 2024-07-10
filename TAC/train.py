import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from common.model import ResNetClassifier
from tac_dataset import TACDataset
from common.dataset import load_dataset, create_dataloaders, split_data
from common.train_utils import (
    get_device,
    get_optimizer,
    get_criterion,
    train_epoch,
    evaluate_model,
)


def train(dataset_type="psa"):
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir / f"models/{dataset_type}"
    models_dir.mkdir(exist_ok=True, parents=True)
    best_model_path = models_dir / "best_model.pth"

    lr = 0.0005
    batch_size = 32
    if dataset_type == "covid":
        num_epochs = 10
        print_epochs = 1
    else:
        num_epochs = 100
        print_epochs = 10

    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(X, y)
    train_dataset = TACDataset(X_train, y_train, label_encoder, transform=None)
    test_dataset = TACDataset(X_test, y_test, label_encoder, transform=None)
    trainloader, testloader = create_dataloaders(
        train_dataset, test_dataset, batch_size
    )

    device = get_device()
    model = ResNetClassifier(channel_in=1, num_classes=num_classes).to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=lr)

    best_f1_score = 0.0

    for epoch in tqdm(range(num_epochs)):
        loss, accuracy = train_epoch(model, trainloader, optimizer, criterion, device)
        if epoch % print_epochs == 0 or epoch == num_epochs - 1:
            y_test_true, y_test_pred = evaluate_model(model, testloader, device)
            val_f1 = f1_score(y_test_true, y_test_pred, average="weighted") * 100

            if val_f1 > best_f1_score:
                best_f1_score = val_f1
                torch.save(model.state_dict(), best_model_path)

            print(f"[{epoch}] loss: {loss:.3f}, val F1-score: {val_f1:.2f}%")

    # Load and evaluate the best model
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
    plt.show()


if __name__ == "__main__":
    # train("psa")
    # train("ddos")
    train("covid")
