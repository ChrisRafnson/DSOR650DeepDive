import torch
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Load a pre-trained model (e.g., ResNet18)
model = models.resnet18(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define the image transformation to match input requirements of the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization used by ImageNet
])

def load_image(image_path):
    """Load and transform an image."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def save_filters(filters, save_dir):
    """Save the filters from a convolutional layer as images."""
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(filters.shape[0]):
        filter_img = filters[i]  # Shape: (C, H, W), where C is the number of input channels

        # If the filter has more than 3 channels, take the first 3 channels for visualization
        if filter_img.shape[0] == 3:
            filter_img = np.transpose(filter_img, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        else:
            # Convert multi-channel filter to grayscale by taking the mean across channels
            filter_img = np.mean(filter_img, axis=0)

        # Normalize filter values to range [0, 255]
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min()) * 255
        filter_img = np.uint8(filter_img)

        # Save the filter as an image
        cmap = "gray" if len(filter_img.shape) == 2 else None  # Use grayscale colormap if 2D
        plt.imsave(f"{save_dir}/filter_{i}.png", filter_img, cmap=cmap)


def save_feature_maps(feature_maps, save_dir):
    """Save the feature maps after passing through the first convolutional layer."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each feature map as an image
    for i in range(feature_maps.shape[0]):
        feature_map = feature_maps[i, :, :]
        plt.imsave(f"{save_dir}/feature_map_{i}.png", feature_map, cmap='viridis')

def visualize_filters(model, save_dir, layer_num=2):
    """Visualize and save the filters from a specified convolutional layer in ResNet-18."""
    if layer_num == 2:
        filters = model.layer1[0].conv1.weight.data.cpu().numpy()  # First conv in layer1
    elif layer_num == 3:
        filters = model.layer1[0].conv2.weight.data.cpu().numpy()  # Second conv in layer1
    elif layer_num == 4:
        filters = model.layer2[0].conv1.weight.data.cpu().numpy()  # First conv in layer2
    else:
        raise ValueError("Currently supports only layers 2, 3, and 4 for visualization.")

    save_filters(filters, save_dir)



def visualize_feature_maps(image, model, save_dir, layer_num=2):
    """Visualize and save the feature maps after passing through the specified convolutional layer."""
    feature_maps = []

    # Hook into the desired layer
    if layer_num == 2:
        hook_layer = model.layer1[0].conv1
    elif layer_num == 3:
        hook_layer = model.layer1[0].conv2
    else:
        raise ValueError("Currently supports only layers 2 and 3 for visualization.")

    def hook_fn(module, input, output):
        feature_maps.append(output.detach())

    hook = hook_layer.register_forward_hook(hook_fn)

    # Pass image through the model to capture the feature maps
    with torch.no_grad():
        model(image)

    hook.remove()  # Remove hook after use

    # Save the feature maps
    feature_maps = feature_maps[0].squeeze(0).cpu().numpy()  # Remove batch dimension
    save_feature_maps(feature_maps, save_dir)


def main():
    # Upload an image
    image_path = "raw_image_files/IMG_4636_Original.JPEG"
    image = load_image(image_path)

    # Specify the directory to save the images
    save_dir = "visualizations_third_layer"

    # Choose the layer to visualize (2 for first conv in layer1, 3 for second conv in layer1)
    layer_num = 3  # Change to 3 for third convolutional layer

    # Visualize and save the filters
    print(f"Saving filters of layer {layer_num}...")
    visualize_filters(model, save_dir, layer_num)

    # Visualize and save the feature maps
    print(f"Saving feature maps after layer {layer_num}...")
    visualize_feature_maps(image, model, save_dir, layer_num)

    print(f"Visualizations saved in '{save_dir}' directory.")

if __name__ == "__main__":
    main()


