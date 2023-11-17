import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModelForSequenceClassification

def load_resnet_model(model_name, num_classes):
    """
    Loads a pre-trained ResNet model and modifies the fully connected layer.

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

    return model

def load_transformer_model(model_name, num_classes):
    """
    Loads a pre-trained Hugging Face transformer model for sequence classification.

    Args:
    - model_name (str): Transformer model name or path.
    - num_classes (int): Number of classes for the sequence classification task.

    Returns:
    - model (transformers.PreTrainedModel): Transformer model for sequence classification.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    return model
