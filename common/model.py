import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=8, image_size=128):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)

        # Reduced pooling size to preserve more information
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.25)

        # Additional convolutional layer to capture more complex patterns
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout(0.25)

        # Batch normalization to stabilize learning
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        fm_size = ((image_size // 2) // 2) // 2  # image_size/2 for each pooling layer

        self.fc1 = nn.Linear(64 * fm_size * fm_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
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
