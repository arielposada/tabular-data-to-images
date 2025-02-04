import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.decomposition import PCA


class DummySelector:
    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def predict(self, X):
        return X


def get_feature_selector(estimator, name="rfecv", args={}):
    if name == "rfecv":
        return RFECV(
            estimator=estimator,
            step=0.1,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring="f1_weighted",
        )
    elif name == "pca":
        return PCA(n_components=args.get("n_components", 22))
    elif name == "mutual_info":
        pass
    elif name == "none":
        return DummySelector()
    else:
        raise ValueError(f"Unknown feature selector: {name}")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(model, lr=1e-4, weight_decay=1e-5):
    return optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )


def get_input_size_and_num_classes(X, y):
    input_size = X.shape[1]
    num_classes = len(set(y))
    return input_size, num_classes


def get_criterion():
    return nn.CrossEntropyLoss()


def train_epoch(model, trainloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for data in trainloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(trainloader), 100 * correct / total


def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return true_labels, predictions
