import torchvision
import torch.nn as nn
import torch

def load_swin():
    model = torchvision.models.swin_t(weights="Swin_T_Weights.IMAGENET1K_V1")
    model = modify_swin_transformer(model)
    model.head = nn.Linear(model.head.in_features, out_features=1)
    return model

def modify_swin_transformer(model):
    original_conv = model.features[0][0]
    NUM_CHANNELS = 24  # Total number of input channels

    # Create a new convolutional layer
    new_conv = nn.Conv2d(NUM_CHANNELS, original_conv.out_channels, 
                        kernel_size=original_conv.kernel_size, 
                        stride=original_conv.stride, 
                        padding=original_conv.padding, 
                        bias=False)

    # Initialize the weights for the new convolutional layer
    with torch.no_grad():
        # Copy weights from the original convolution for the first 12 channels
        for i in range(0, 12, 3):
            new_conv.weight[:, i:i+3, :, :] = original_conv.weight.clone()

        # Calculate the mean of the original weights and repeat them for the remaining channels
        mean_rgb_weights = original_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 12:, :, :] = mean_rgb_weights.repeat(1, NUM_CHANNELS - 12, 1, 1)

        # Scale weights to maintain similar activation magnitudes
        new_conv.weight.data *= (3 / NUM_CHANNELS)**0.5

    # Replace the original convolutional layer with the new one
    model.features[0][0] = new_conv

    return model