import torchvision
import torch

def load_swin3d():
    model = torchvision.models.video.swin3d_b(weights="Swin3D_B_Weights.KINETICS400_IMAGENET22K_V1")

    # Access the first Conv3d layer which is part of the PatchEmbed3d module
    first_conv_layer = model.patch_embed.proj

    # Create a new Conv3d layer with 6 input channels
    new_first_conv = torch.nn.Conv3d(
        in_channels=6,  # Increase from 3 to 6
        out_channels=first_conv_layer.out_channels,
        kernel_size=first_conv_layer.kernel_size,
        stride=first_conv_layer.stride,
        padding=first_conv_layer.padding,
        bias=first_conv_layer.bias is not None
    )

    # Initialize the weights for the new Conv3d layer
    # One common method is to average the original RGB weights and replicate them for the additional channels
    with torch.no_grad():
        original_weights = first_conv_layer.weight
        # Extend the weights by repeating the mean of the original three channels
        new_weights = torch.cat([original_weights, original_weights.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1, 1)], dim=1)
        new_first_conv.weight.data = new_weights
        model.head = torch.nn.Linear(in_features=1024, out_features=1, bias=True)

    # Replace the original first convolutional layer with the new one
    model.patch_embed.proj = new_first_conv

    return model