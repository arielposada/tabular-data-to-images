from pathlib import Path
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the transformation pipeline
transform = transforms.Compose(
    [
        # transforms.Resize(256),  # Resize the image to 256x256
        # transforms.CenterCrop(224),  # Center crop the image to 224x224
        transforms.Resize(36),
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        # transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
    ]
)


def load_and_process_image(image_path):
    """
    Load an image, center crop to 224x224, and normalize the pixel values to [-1, 1].

    :param image_path: Path to the image file.
    :return: Transformed image tensor.
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Apply the transformations
    transformed_image = transform(image)

    return transformed_image


if __name__ == "__main__":
    # Example usage
    cur_dir = Path(__file__).parent.absolute()
    image_path = cur_dir / "images/maltese_dog.jpg"
    processed_image = load_and_process_image(image_path)
    print(processed_image.shape)  # Should print torch.Size([3, 224, 224])

    # Plot the processed image
    plt.imshow(processed_image.permute(1, 2, 0))
    plt.axis("off")
    plt.show()
