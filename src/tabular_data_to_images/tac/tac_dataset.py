from pathlib import Path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from tabular_data_to_images.tac.feature_vector_transformer import (
    FeatureVectorTransformer,
)
from tabular_data_to_images.tac.load_image import load_and_process_image
from tabular_data_to_images.common.dataset import (
    load_psa_data,
    split_data,
    create_dataloaders,
)


class TACDataset(Dataset):
    def __init__(self, X, y, label_encoder, feature_vector_transformer, transform=None):
        self.X = X
        self.y = y
        self.label_encoder = label_encoder
        self.transform = transform

        # Initialize the feature vector transformer
        self.feature_vector_transformer = feature_vector_transformer
        self.base_image = self.load_base_image()

    def load_base_image(self):
        cur_dir = Path(__file__).parent.absolute()
        image_path = cur_dir / "images/chess_pattern_36.jpg"
        return load_and_process_image(image_path).unsqueeze(0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        feature_vector = self.X[idx]
        kernel = self.feature_vector_transformer.transform([feature_vector])[0]
        kernel_tensor = (
            torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )  # Add channel and batch dimensions

        # Apply the kernel to the base image using convolution
        with torch.no_grad():
            padding = (kernel_tensor.shape[0] - 1) // 2
            convolved_image = F.conv2d(
                self.base_image, kernel_tensor, padding=padding
            ).squeeze(0)

        if self.transform:
            convolved_image = self.transform(convolved_image)

        label = self.y[idx]
        return convolved_image, label


def main():
    # Load and split the data
    X, y = load_psa_data()
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(X, y)

    # Create train and test datasets
    train_dataset = TACDataset(X_train, y_train, label_encoder, transform=None)
    test_dataset = TACDataset(X_test, y_test, label_encoder, transform=None)

    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset)

    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break


if __name__ == "__main__":
    main()
