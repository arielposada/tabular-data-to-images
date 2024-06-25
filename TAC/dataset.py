from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from feature_vector_transformer import FeatureVectorTransformer
from load_image import load_and_process_image


class TACDataset(Dataset):
    def __init__(self, X, y, label_encoder, transform=None):
        self.X = X
        self.y = y
        self.label_encoder = label_encoder
        self.transform = transform

        # Initialize the feature vector transformer
        self.feature_vector_transformer = FeatureVectorTransformer()
        self.feature_vector_transformer.fit(self.X)
        self.base_image = self.load_base_image()

    def load_base_image(self):
        cur_dir = Path(__file__).parent.absolute()
        image_path = cur_dir / "images/maltese_dog.jpg"
        return load_and_process_image(image_path)

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
            convolved_image = F.conv2d(
                self.base_image.unsqueeze(0), kernel_tensor, padding="same"
            ).squeeze(0)

        if self.transform:
            convolved_image = self.transform(convolved_image)

        label = self.y[idx]
        return convolved_image, label


def load_psa_data():
    df = pd.read_csv("dataset/psa_spectral_ratio.csv")
    labels = pd.read_csv("dataset/psa_labels.csv")[["file", "soil_type"]]

    df = df.merge(labels, left_on="ext_station_key", right_on="file", how="left")
    df = df[df["orientation"] == "H/V"]
    df = df.drop(columns=["station_key", "index", "orientation", "file"]).set_index(
        ["ext_station_key"]
    )

    X = df.drop(columns=["soil_type"]).values
    y = df["soil_type"].values
    return X, y


def split_data(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y
    )

    num_classes = np.unique(y_encoded).size

    return X_train, X_test, y_train, y_test, le, num_classes


def create_dataloaders(train_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


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
