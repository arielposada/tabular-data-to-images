import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1].as_posix())
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from common.train_utils import (
    evaluate_model,
    get_criterion,
    get_device,
    get_optimizer,
    train_epoch,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)
from common.dataset import create_dataloaders, load_dataset, split_data
from baselines.sequential_model import create_model
from torch.utils.data import TensorDataset


def train_model(model_name="transformer", dataset_type="psa"):
    print(f"Training {model_name} model on {dataset_type} dataset...")
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir / f"models/{dataset_type}"
    models_dir.mkdir(exist_ok=True, parents=True)
    best_model_path = models_dir / f"best_model.pth"

    # Hyperparameters
    lr = 0.0005
    if dataset_type == "ddos":
        lr = 1e-4
    elif dataset_type == "covid":
        num_epochs = 10
        print_epochs = 1
    else:
        num_epochs = 100
        print_epochs = 10

    # Setup model and training essentials
    batch_size = 32
    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
        X, y, return_tensor=True
    )
    # Expand dimensions to hava bs, seq_len, input_size
    X_train = X_train.unsqueeze(-1)
    X_test = X_test.unsqueeze(-1)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    trainloader, testloader = create_dataloaders(
        train_dataset, test_dataset, batch_size=batch_size
    )

    seq_len = X_train.shape[1]
    device = get_device()
    model = create_model(
        model_name=model_name,
        input_size=1,
        seq_len=seq_len,
        num_classes=num_classes,
        device=device,
    )
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
    train_model("transformer", "psa")
    # train_model("lstm", "psa")
