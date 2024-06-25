from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import pandas as pd
from model import ResNetClassifier
from dataset import TACDataset, load_psa_data, create_dataloaders, split_data

# Load and split the data
X, y = load_psa_data()
X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(X, y)

# Create train and test datasets
train_dataset = TACDataset(X_train, y_train, label_encoder, transform=None)
test_dataset = TACDataset(X_test, y_test, label_encoder, transform=None)

# Create DataLoaders
trainloader, testloader = create_dataloaders(train_dataset, test_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32


model = ResNetClassifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5
)

batch_size = 32
num_classes = 8  # Adjust this according to your needs
model.train()
for epoch in range(100):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # accumulate loss
        running_loss += loss.item()

        # calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # print epoch statistics
    if epoch % 20 == 0:
        print(
            f"[{epoch}] loss: {running_loss / len(trainloader):.3f}, accuracy: {100 * correct / total:.2f}%"
        )

print(
    f"Final epoch [{epoch}] loss: {running_loss / len(trainloader):.3f}, accuracy: {100 * correct / total:.2f}%"
)


# Evaluate the model
model.to("cpu")
model.eval()
y_train_pred, y_test_pred = [], []
y_train_true, y_test_true = [], []

with torch.no_grad():
    for data in trainloader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_train_pred.extend(predicted.numpy())
        y_train_true.extend(labels.numpy())

    for data in testloader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_test_pred.extend(predicted.numpy())
        y_test_true.extend(labels.numpy())

# Calculate accuracy
train_accuracy = accuracy_score(y_train_true, y_train_pred)
test_accuracy = accuracy_score(y_test_true, y_test_pred)

print(f"The train accuracy was {train_accuracy:.3f}")
print(f"The test accuracy was {test_accuracy:.3f}")

# Display confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test_true, y_test_pred, display_labels=label_encoder.classes_
)
plt.show()
