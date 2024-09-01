import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt


class VectorToImageNetwork(nn.Module):
    def __init__(self, input_size=16):
        super(VectorToImageNetwork, self).__init__()
        closest_square = int(
            torch.round(torch.sqrt(torch.tensor(input_size, dtype=torch.float32)))
        )

        # Fully connected layer to reduce dimensionality
        self.fc1 = nn.Linear(input_size, closest_square * closest_square)

        # Upconvolution (Transposed Convolution) layers
        self.upconv1 = nn.ConvTranspose2d(
            1, 16, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.upconv2 = nn.ConvTranspose2d(
            16, 8, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.upconv3 = nn.ConvTranspose2d(
            8, 3, kernel_size=3, stride=2, padding=1, output_padding=1
        )

    def forward(self, x):
        x = self.fc1(x)
        closest_square = int(
            torch.round(torch.sqrt(torch.tensor(x.size(1), dtype=torch.float32)))
        )
        x = x.view(-1, 1, closest_square, closest_square)

        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = torch.sigmoid(self.upconv3(x))  # Float image [0, 1]
        return x

    def calculate_output_size(
        self, input_size, kernel_size, stride, padding, output_padding
    ):
        return (input_size - 1) * stride - 2 * padding + kernel_size + output_padding

    def get_output_image_size(self, input_size):
        closest_square = int(
            torch.round(torch.sqrt(torch.tensor(input_size, dtype=torch.float32)))
        )
        size_after_fc1 = closest_square

        size_after_upconv1 = self.calculate_output_size(size_after_fc1, 3, 2, 1, 1)
        size_after_upconv2 = self.calculate_output_size(size_after_upconv1, 3, 2, 1, 1)
        size_after_upconv3 = self.calculate_output_size(size_after_upconv2, 3, 2, 1, 1)

        return size_after_upconv3


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8, input_image_size=112):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(
            32 * (input_image_size // 4) * (input_image_size // 4), 128
        )  # Adjust the size according to your input
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Adjust the size according to your input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CombinedModel(nn.Module):
    def __init__(self, input_size=16, num_classes=8, use_resnet=False):
        super(CombinedModel, self).__init__()
        self.vector_to_image = VectorToImageNetwork(input_size)
        image_size = self.vector_to_image.get_output_image_size(input_size)

        if use_resnet:
            self.cnn = models.resnet18(pretrained=True)
            # Freeze all layers
            for param in self.cnn.parameters():
                param.requires_grad = False
            # Replace the fully connected layer
            self.cnn.fc = nn.Linear(self.cnn.fc.in_features, num_classes)
            # Define the resize transform
            self.resize_transform = transforms.Resize((224, 224))
            self.normalize_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            self.cnn = SimpleCNN(num_classes=num_classes, input_image_size=image_size)

    def forward(self, x):
        x = self.vector_to_image(x)
        if hasattr(self, "resize_transform"):
            x = self.resize_transform(x)
        if hasattr(self, "normalize_transform"):
            x = self.normalize_transform(x)
        x = self.cnn(x)
        return x

    def get_image_representation(self, x):
        return self.vector_to_image(x)


if __name__ == "__main__":
    # Example usage
    input_size = 100
    num_classes = 8

    # Option to use ResNet-18
    use_resnet = True
    model = CombinedModel(
        input_size=input_size, num_classes=num_classes, use_resnet=use_resnet
    )
    print(model)

    # Generate a random input tensor with batch size of 4 and input size of 16
    input_tensor = torch.randn(32, input_size)

    # Forward pass
    output = model(input_tensor)
    print(output)

    # Visualize image representation
    model.visualize_image_representation(input_tensor)
