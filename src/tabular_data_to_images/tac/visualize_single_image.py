import torch
import numpy as np

from tabular_data_to_images.tac.feature_vector_transformer import (
    FeatureVectorTransformer,
)
from tabular_data_to_images.tac.tac_dataset import TACDataset
from tabular_data_to_images.common.dataset import (
    create_dataloaders,
    load_dataset,
    split_data,
)
from matplotlib import pyplot as plt


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
            # Find the index of the image with the highest variance
            most_different_idx = np.argmax(variances)
            # Use this image as the representative image
            img = class_images_np[most_different_idx]
            img = (img - img_min) / (img_max - img_min)
            class_images.append(img)
        else:
            # If only one image in the class, just use that image
            img = class_images_np[0]
            img = (img - img_min) / (img_max - img_min)
            class_images.append(img)

    return class_images


dataset_type = "ddos"
X, y = load_dataset(dataset_type)
X_train, X_test, y_train, y_test, label_encoder, num_classes = split_data(X, y)
feature_vector_transformer = FeatureVectorTransformer()
feature_vector_transformer.fit(X_train)
train_dataset = TACDataset(
    X_train,
    y_train,
    label_encoder,
    feature_vector_transformer=feature_vector_transformer,
    transform=None,
)
test_dataset = TACDataset(
    X_test,
    y_test,
    label_encoder,
    feature_vector_transformer=feature_vector_transformer,
    transform=None,
)
trainloader, testloader = create_dataloaders(train_dataset, test_dataset)


# Load and display the base image
base_image = train_dataset.load_base_image()
base_image = base_image.squeeze(0).permute(1, 2, 0).numpy()

# Collect all inputs and labels from the test loader
all_inputs = []
all_labels = []
for data in testloader:
    inputs, labels = data
    all_inputs.append(inputs)
    all_labels.append(labels)
all_inputs = torch.cat(all_inputs)
all_labels = torch.cat(all_labels)

# Get class names from label encoder
class_names = label_encoder.classes_

# Get one image per class
class_images = get_one_image_per_class(all_inputs, all_labels, class_names)

# Plot the base image and one image per class side by side
fig, axes = plt.subplots(1, num_classes + 1, figsize=(15, 5))
axes[0].imshow(base_image, cmap="gray")
axes[0].set_title("Base Image")
axes[0].axis("off")

for i in range(num_classes):
    axes[i + 1].imshow(class_images[i], cmap="gray")
    axes[i + 1].set_title(class_names[i])
    axes[i + 1].axis("off")

plt.tight_layout()
plt.show()
