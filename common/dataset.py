import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torch


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


def split_data(X, y, return_tensor: bool = False):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y
    )

    num_classes = np.unique(y_encoded).size

    if return_tensor:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
    return X_train, X_test, y_train, y_test, le, num_classes


def create_dataloaders(train_dataset, test_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
