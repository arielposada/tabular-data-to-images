import torch
import torch.optim as optim
import torch.nn as nn


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


def evaluate_model(model, trainloader, testloader, device):
    model.to(device)
    model.eval()
    y_train_pred, y_test_pred = [], []
    y_train_true, y_test_true = [], []

    with torch.no_grad():
        for data in trainloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_train_pred.extend(predicted.cpu().numpy())
            y_train_true.extend(labels.cpu().numpy())

        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_test_pred.extend(predicted.cpu().numpy())
            y_test_true.extend(labels.cpu().numpy())

    return y_train_true, y_train_pred, y_test_true, y_test_pred
