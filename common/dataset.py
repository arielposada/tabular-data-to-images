import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torch
import sys
from pathlib import Path
import pandas as pd

sys.path.append(Path(__file__).resolve().parents[1].as_posix())


from tabgan.sampler import (
    GANGenerator,
    LLMGenerator,
)

from ForestDiffusion import ForestDiffusionModel


dataset_dir = Path(__file__).resolve().parents[1] / "dataset"


def augment_data(X_train, y_train, X_test, augment_strategy="original", return_np=True):
    if augment_strategy == "none":
        return X_train, y_train

    # Convert to pandas dataframe
    X_train = pd.DataFrame(
        X_train, columns=["f" + str(i) for i in range(X_train.shape[1])]
    )
    X_test = pd.DataFrame(
        X_test, columns=["f" + str(i) for i in range(X_test.shape[1])]
    )
    y_train = pd.DataFrame(y_train, columns=["Label"])
    print(f"Training data before augmentation - shape: {X_train.shape}")
    print(f"Augmentation strategy: {augment_strategy}")
    if augment_strategy == "gan":
        X_gen, y_gen = GANGenerator(
            gen_params={"batch_size": 500, "epochs": 10, "patience": 5}
        ).generate_data_pipe(
            X_train,
            y_train,
            X_test,
        )
        X_train = pd.concat([X_train, X_gen], axis=0)
        y_train = pd.concat([y_train, y_gen.to_frame()], axis=0)
    elif augment_strategy == "forest_diffusion":
        forest_model = ForestDiffusionModel(
            X=X_train.to_numpy(),
            label_y=y_train["Label"].to_numpy(),
            n_t=50,
            duplicate_K=100,
            n_jobs=-1,
        )
        Xy_fake = forest_model.generate(batch_size=int(X_train.shape[0] * 1.1))
        # Separate the labels and train generated data
        X_gen = Xy_fake[:, :-1]
        y_gen = Xy_fake[:, -1]
        X_train = pd.concat([X_train, pd.DataFrame(X_gen)], axis=0)
        y_train = pd.concat([y_train, pd.DataFrame(y_gen)], axis=0)
    elif augment_strategy == "llm":
        new_train, new_target = LLMGenerator(
            gen_params={
                "batch_size": 2,
                "epochs": 4,
                # "llm": "distilgpt2",
                "llm": "facebook/opt-125m",
                "max_length": 500,
            }
        ).generate_data_pipe(
            X_train,
            y_train,
            X_test,
        )
        X_train = new_train
        y_train = new_target

    print(f"Training data after augmentation - shape: {X_train.shape}")
    if return_np:
        return X_train.values, y_train["Label"].values
    return X_train, y_train


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
    df = pd.read_csv(dataset_dir / "covid/dataset_nuevo_corregido.csv")
    y = df["Risk_Label"].values
    X = df.drop(columns=["Risk_Label", "Risk"]).values
    return X, y


def load_ddos_data():
    df = pd.read_csv(dataset_dir / "DDoS/dataset_two_classes.csv")
    y = df["Label"].values
    X = df.drop(columns=["Label"])
    numeric_columns = X.select_dtypes(include=["number"]).columns
    X_numeric = X[numeric_columns].values
    return X_numeric, y


def load_psa_data():
    df = pd.read_csv(dataset_dir / "earthquake/psa_spectral_ratio.csv")
    labels = pd.read_csv(dataset_dir / "earthquake/psa_labels.csv")[
        ["file", "soil_type"]
    ]

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
