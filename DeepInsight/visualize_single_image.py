import sys
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from DeepInsight.train import generate_tensor_data
from common.dataset import load_dataset, split_data
from torch.utils.data import TensorDataset


def get_one_image_per_class(images, labels, class_names):
    num_classes = len(class_names)
    class_images = []
    img_max = images.max().cpu().numpy()
    img_min = images.min().cpu().numpy()
    for i in range(num_classes):
        class_indices = (labels == i).nonzero(as_tuple=True)[0]
        if len(class_indices) > 0:
            class_images_subset = images[class_indices]
            class_images_np = class_images_subset.permute(0, 2, 3, 1).cpu().numpy()
            variances = np.var(class_images_np, axis=(1, 2, 3))
            most_different_idx = np.argmax(variances)
            img = class_images_np[most_different_idx]
            img = (img - img_min) / (img_max - img_min)
            class_images.append(img)
        else:
            img = class_images_np[0]
            img = (img - img_min) / (img_max - img_min)
            class_images.append(img)

    return class_images


def visualize_class_images(dataset_type):
    X, y = load_dataset(dataset_type)
    X_train, _, y_train, _, label_encoder, num_classes = split_data(X, y)

    X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(X, y)
    X_train, X_test, y_train, y_test = generate_tensor_data(
        X_train, X_test, y_train, y_test
    )

    train_dataset = TensorDataset(X_train, y_train)

    all_inputs = []
    all_labels = []
    for data in train_dataset:
        inputs, labels = data
        all_inputs.append(inputs)
        all_labels.append(labels)
    all_inputs = torch.stack(all_inputs)
    all_labels = torch.stack(all_labels)

    class_images = get_one_image_per_class(
        all_inputs, all_labels, label_encoder.classes_
    )

    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))
    for i in range(num_classes):
        axes[i].imshow(class_images[i], cmap="gray")
        axes[i].set_title(label_encoder.classes_[i])
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset_type = "ddos"  # Change to your dataset type if needed
    visualize_class_images(dataset_type)
