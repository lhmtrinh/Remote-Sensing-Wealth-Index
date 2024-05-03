import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torchvision import transforms

NUM_CHANNELS = 24

def load_resnet_model(num_classes=1):
    """
    Loads a pre-trained ResNet model and modifies the fully connected layer and conv1 layer to accept more channels.

    Args:
    - model_name (str): Name of the ResNet model to load.
    - num_classes (int): Number of classes for the new fully connected layer.

    Returns:
    - model (torch.nn.Module): Modified ResNet model.
    """
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    
    # Modify the fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Modify resnet to include more input channels
    model = modify_resnet_model(model)

    return model


def modify_resnet_model(model):
    """
    Modifies a ResNet model to accept a different number of input channels
    and initializes the weights of the first convolutional layer explicitly.
    RGB chanels would have the same weight as the original resnet. Non RGB channels
    are initialized to the mean weight of RGB channels. All weights are scaled to 3/C 
    because of same scale initilization.

    Args:
    - model (torch.nn.Module): Original ResNet model.
    - num_input_channels (int): Number of input channels for the new model.

    Returns:
    - torch.nn.Module: Modified ResNet model.
    """
    original_conv1 = model.conv1
    new_conv1 = nn.Conv2d(NUM_CHANNELS, original_conv1.out_channels, 
                          kernel_size=original_conv1.kernel_size, 
                          stride=original_conv1.stride, 
                          padding=original_conv1.padding, 
                          bias=False)

    # Initialize the weights explicitly for the RGB channels with pre-trained ImageNet weights
    with torch.no_grad():
        # First set of RGB channels
        new_conv1.weight[:, 0:3, :, :] = original_conv1.weight.clone()
        # Second set of RGB channels
        new_conv1.weight[:, 3:6, :, :] = original_conv1.weight.clone()
        # Third set of RGB channels
        new_conv1.weight[:, 6:9, :, :] = original_conv1.weight.clone()
        # Fourth set of RGB channels
        new_conv1.weight[:, 9:12, :, :] = original_conv1.weight.clone()

        # Initialize the weights for the non-RGB channels as the mean of the RGB channel weights
        mean_rgb_weights = original_conv1.weight.mean(dim=1, keepdim=True)
        new_conv1.weight[:, 12:, :, :] = mean_rgb_weights.repeat(1, NUM_CHANNELS - 12, 1, 1)

        # Scale all weights by 3/C
        new_conv1.weight.data *= (3 / NUM_CHANNELS)**0.5

    model.conv1 = new_conv1

    return model

def register_model_with_hook(model):    
    # Define a forward hook to capture the outputs of the last conv layer
    feature_maps = None
    def hook(module, input, output):
        nonlocal feature_maps
        feature_maps = output
    
    # Register the hook to the last layer in `layer4`
    model.layer4[-1].register_forward_hook(hook)

    # Return both the model and a function to get feature maps
    def get_feature_maps(x):
        nonlocal feature_maps
        _ = model(x)  # Perform a forward pass to fill in `feature_maps`
        return feature_maps

    return get_feature_maps


