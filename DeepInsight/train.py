import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from common.dataset import augment_data, load_dataset, create_dataloaders, split_data
from common.train_utils import (
    get_device,
    get_optimizer,
    get_criterion,
    train_epoch,
    evaluate_model,
)
from pyDeepInsight import ImageTransformer
from pyDeepInsight.utils import Norm2Scaler
from sklearn.manifold import TSNE
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from common.model import SimpleCNN


def generate_tensor_data(X_train, X_test, y_train, y_test):
    # Normalization step
    ln = Norm2Scaler()
    X_train_norm = ln.fit_transform(X_train)
    X_test_norm = ln.transform(X_test)

    # TSNE transformation
    perplexity = np.sqrt(X_train_norm.shape[1])
    perplexity = min(max(perplexity, 5), 50)
    distance_metric = "cosine"
    reducer = TSNE(
        n_components=2,
        metric=distance_metric,
        init="random",
        learning_rate="auto",
        perplexity=perplexity,
        n_jobs=-1,
    )

    # Building the model
    image_size = (36, 36)
    it = ImageTransformer(feature_extractor=reducer, pixels=image_size)
    it.fit(X_train_norm, y=y_train, plot=True)
    X_train_img = it.transform(X_train_norm)
    X_test_img = it.transform(X_test_norm)
    print("Finished fitting image transformer...")

    preprocess = transforms.Compose([transforms.ToTensor()])
    X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float()
    y_train_tensor = torch.from_numpy(y_train)
    print("Finished transforming training data...")

    X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float()
    y_test_tensor = torch.from_numpy(y_test)
    print("Finished transforming test data...")
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def train(dataset_type="psa", augment_strategy="none"):
    current_dir = Path(__file__).resolve().parent
    models_dir = current_dir / f"models/{dataset_type}"
    models_dir.mkdir(exist_ok=True, parents=True)
    best_model_path = models_dir / "best_model.pth"

    lr = 1e-3
    batch_size = 32
    if dataset_type == "covid":
        num_epochs = 10
        print_epochs = 1
    else:
        num_epochs = 200
        print_epochs = 10

    X, y = load_dataset(dataset_type)
    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(X, y)

    X_train, y_train = augment_data(
        X_train, y_train, X_test, augment_strategy, return_np=True
    )
    X_train, X_test, y_train, y_test = generate_tensor_data(
        X_train, X_test, y_train, y_test
    )

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    trainloader, testloader = create_dataloaders(
        train_dataset, test_dataset, batch_size
    )

    device = get_device()
    # model = ResNetClassifier(channel_in=3, num_classes=num_classes).to(device)
    model = SimpleCNN(
        in_channels=3,
        num_classes=num_classes,
        image_size=36,
    ).to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(model, lr=lr)

    best_f1_score = 0.0

    for epoch in tqdm(range(num_epochs)):
        loss, accuracy = train_epoch(model, trainloader, optimizer, criterion, device)
        if epoch % print_epochs == 0 or epoch == num_epochs - 1:
            y_test_true, y_test_pred = evaluate_model(model, testloader, device)
            val_f1 = f1_score(y_test_true, y_test_pred, average="weighted") * 100

            if val_f1 > best_f1_score:
                best_f1_score = val_f1
                torch.save(model.state_dict(), best_model_path)

            print(f"[{epoch}] loss: {loss:.3f}, val F1-score: {val_f1:.2f}%")

    # Load and evaluate the best model
    model.load_state_dict(torch.load(best_model_path))
    y_test_true, y_test_pred = evaluate_model(model, testloader, device)
    test_f1 = f1_score(y_test_true, y_test_pred, average="weighted")
    test_accuracy = accuracy_score(y_test_true, y_test_pred)
    test_precision = precision_score(y_test_true, y_test_pred, average="weighted")
    test_recall = recall_score(y_test_true, y_test_pred, average="weighted")

    print(f"Test F1-score: {test_f1 * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Precision: {test_precision * 100:.2f}%")
    print(f"Test Recall: {test_recall * 100:.2f}%")

    ConfusionMatrixDisplay.from_predictions(
        y_test_true, y_test_pred, display_labels=label_encoder.classes_
    )
    plt.show()


if __name__ == "__main__":
    for augment_strategy in [
        "none",
        "gan",
    ]:
        train("ddos", augment_strategy)
