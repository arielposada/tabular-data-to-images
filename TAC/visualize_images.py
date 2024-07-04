import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

from TAC.tac_dataset import TACDataset

import torch
from common.dataset import create_dataloaders, load_psa_data, split_data
from matplotlib import pyplot as plt

current_dir = Path(__file__).resolve().parent
models_dir = current_dir / "models"
best_model_path = models_dir / "best_model.pth"


def visualize_samples_with_labels(images, labels, class_names, num_samples=5):
    num_classes = len(class_names)

    # Create a figure to hold the subplots
    fig, axes = plt.subplots(num_classes, num_samples, figsize=(15, num_classes * 3))

    # Loop over each class
    for i in range(num_classes):
        class_indices = (labels == i).nonzero(as_tuple=True)[0]
        class_images = images[class_indices]

        # Sample up to num_samples images for the class
        num_images_to_display = min(len(class_images), num_samples)
        sampled_images = class_images[:num_images_to_display]

        for j in range(num_images_to_display):
            img = sampled_images[j].permute(1, 2, 0).numpy()
            ax = axes[i, j]
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                axes[i, 0].text(
                    0.5,
                    0.5,
                    class_names[i],
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontsize=12,
                )
                axes[i, 0].axis("off")
    plt.tight_layout()
    plt.show()


X, y = load_psa_data()
X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(X, y)
train_dataset = TACDataset(X_train, y_train, label_encoder, transform=None)
test_dataset = TACDataset(X_test, y_test, label_encoder, transform=None)
trainloader, testloader = create_dataloaders(train_dataset, test_dataset)

# Get class names from label encoder
class_names = label_encoder.classes_

# Visualize the inputs (images) with labels
all_inputs = []
all_labels = []

# Collect all inputs and labels from the test loader
for data in testloader:
    inputs, labels = data
    all_inputs.append(inputs)
    all_labels.append(labels)

all_inputs = torch.cat(all_inputs)
all_labels = torch.cat(all_labels)

# Visualize samples with labels
visualize_samples_with_labels(all_inputs, all_labels, class_names, num_samples=5)
