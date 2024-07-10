import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Output size: 16x15x15
        self.dropout1 = nn.Dropout(0.25)  # Dropout added for regularization
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Output size: 32x7x7
        self.dropout2 = nn.Dropout(0.25)  # Another dropout layer for regularization
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ResNetClassifier(nn.Module):
    def __init__(self, channel_in=3, num_classes=8):
        super(ResNetClassifier, self).__init__()
        # Load the pre-trained ResNet18 model
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        if channel_in != 3:
            # Modify the first convolutional layer to accept `channel_in` channels
            self.resnet.conv1 = nn.Conv2d(
                channel_in, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # Freeze all layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer with a new layer that has `num_classes` outputs
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        # Only unfreeze the fully connected layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)
