import torch
import torch.nn as nn
from torchvision import models

def load_resnet_model(model_name, num_classes, num_input_channels= 24):
    """
    Loads a pre-trained ResNet model and modifies the fully connected layer and conv1 layer to accept more channels.

    Args:
    - model_name (str): Name of the ResNet model to load.
    - num_classes (int): Number of classes for the new fully connected layer.

    Returns:
    - model (torch.nn.Module): Modified ResNet model.
    """
    # Load the pre-trained ResNet model
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError("Unsupported ResNet model: {}".format(model_name))

    # Modify the fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Modify resnet to include more input channels
    model = modify_resnet_model(model, num_input_channels)

    return model

def modify_resnet_model(model, num_input_channels):
    # Load a pre-trained ResNet model
    original_conv1 = model.conv1

    # Create a new Conv2d layer with 24 input channels
    # but the same output channels, kernel size, stride, padding, etc., as the original conv1
    model.conv1 = nn.Conv2d(num_input_channels, original_conv1.out_channels, 
                          kernel_size=original_conv1.kernel_size, 
                          stride=original_conv1.stride, 
                          padding=original_conv1.padding, 
                          bias=False)
    
    # Initialize the weights for the new conv1 layer
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')

    return model