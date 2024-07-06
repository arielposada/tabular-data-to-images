import sys
from pathlib import Path

import torch

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from model import ResNetClassifier
from tac_dataset import TACDataset
from common.dataset import load_psa_data, create_dataloaders, split_data
from common.train_utils import (
    get_device,
    get_optimizer,
    get_criterion,
    train_epoch,
    evaluate_model,
)

current_dir = Path(__file__).resolve().parent
models_dir = current_dir / "models"
models_dir.mkdir(exist_ok=True)
best_model_path = models_dir / "best_model.pth"

lr = 0.0005
num_epochs = 200
batch_size = 32

X, y = load_psa_data()
X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(X, y)
train_dataset = TACDataset(X_train, y_train, label_encoder, transform=None)
test_dataset = TACDataset(X_test, y_test, label_encoder, transform=None)
trainloader, testloader = create_dataloaders(train_dataset, test_dataset)

device = get_device()
model = ResNetClassifier(num_classes=num_classes).to(device)
criterion = get_criterion()
optimizer = get_optimizer(model, lr=lr)

best_f1_score = 0.0

for epoch in range(num_epochs):
    loss, accuracy = train_epoch(model, trainloader, optimizer, criterion, device)
    if epoch % 20 == 0 or epoch == num_epochs - 1:
        y_train_true, y_train_pred, y_test_true, y_test_pred = evaluate_model(
            model, trainloader, testloader, device
        )
        val_f1 = f1_score(y_test_true, y_test_pred, average="weighted") * 100

        if val_f1 > best_f1_score:
            best_f1_score = val_f1
            torch.save(model.state_dict(), best_model_path)

        print(f"[{epoch}] loss: {loss:.3f}, val F1-score: {val_f1:.2f}%")

print(f"Best F1-score: {best_f1_score:.2f}%")
model.load_state_dict(torch.load(best_model_path))

ConfusionMatrixDisplay.from_predictions(
    y_test_true, y_test_pred, display_labels=label_encoder.classes_
)
plt.show()
