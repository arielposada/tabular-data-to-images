import torch
from torch.utils.data import TensorDataset

from tabular_data_to_images.common.dataset import (
    create_dataloaders,
    load_psa_data,
    split_data,
)
from tabular_data_to_images.upsampling_approach.model import CombinedModel
from matplotlib import pyplot as plt
from tabular_data_to_images.common.train_utils import get_device
from tabular_data_to_images.paths import MODELS_DIR

models_dir = MODELS_DIR / "upsampling_approach" / "psa"
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


# Load the data and model
X, y = load_psa_data()
X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(
    X, y, return_tensor=True
)
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
trainloader, testloader = create_dataloaders(train_dataset, test_dataset)
input_size = X_train.shape[1]

device = get_device()
model = CombinedModel(input_size, num_classes).to(device)
model.load_state_dict(torch.load(best_model_path))

# Get class names from label encoder
class_names = label_encoder.classes_

# Visualize the image representations
model.eval()
with torch.no_grad():
    all_inputs = []
    all_labels = []

    # Collect all inputs and labels from the test loader
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device).long()
        all_inputs.append(inputs)
        all_labels.append(labels)

    all_inputs = torch.cat(all_inputs)
    all_labels = torch.cat(all_labels)

    # Get image representations
    image_reps = model.get_image_representation(all_inputs).detach().cpu()
    all_labels = all_labels.cpu()

    # Visualize samples with labels
    visualize_samples_with_labels(image_reps, all_labels, class_names, num_samples=5)
