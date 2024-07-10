import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torch


def load_dataset(dataset_type):
    if dataset_type == "psa":
        return load_psa_data()
    elif dataset_type == "ddos":
        return load_ddos_data()
    elif dataset_type == "covid":
        return load_covid_data()
    else:
        raise ValueError("Unsupported dataset type")


def load_covid_data():
    df = pd.read_csv("dataset/covid/dataset_nuevo_corregido.csv")
    y = df["Risk_Label"].values
    X = df.drop(columns=["Risk_Label", "Risk"]).values
    return X, y


def load_ddos_data():
    df = pd.read_csv("dataset/DDoS/dataset_two_classes.csv")
    y = df["Label"].values
    X = df.drop(columns=["Label"])
    numeric_columns = X.select_dtypes(include=["number"]).columns
    X_numeric = X[numeric_columns].values
    return X_numeric, y


def load_psa_data():
    df = pd.read_csv("dataset/earthquake/psa_spectral_ratio.csv")
    labels = pd.read_csv("dataset/earthquake/psa_labels.csv")[["file", "soil_type"]]

    df = df.merge(labels, left_on="ext_station_key", right_on="file", how="left")
    df = df[df["orientation"] == "H/V"]
    df = df.drop(columns=["station_key", "index", "orientation", "file"]).set_index(
        ["ext_station_key"]
    )

    X = df.drop(columns=["soil_type"]).values
    y = df["soil_type"].values
    return X, y


def split_data(
    X,
    y,
    return_tensor: bool = False,
    random_state: int = 42,
    use_statify: bool = True,
):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    stratify = None
    if use_statify:
        stratify = y_encoded

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=random_state,
        stratify=stratify,
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
