import torch
import torch.nn as nn
from torchvision import models

NUM_CHANNELS = 6
def initialize_resnet(model_name):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError("Unsupported ResNet model: {}".format(model_name))

    # Modify the first convolutional layer
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(NUM_CHANNELS, original_conv1.out_channels, 
                            kernel_size=original_conv1.kernel_size, 
                            stride=original_conv1.stride, 
                            padding=original_conv1.padding, 
                            bias=False)
    
    # Initialize the weights
    with torch.no_grad():
        # Copy the weights for RGB channels
        model.conv1.weight[:, :3, :, :] = original_conv1.weight.clone()
        # Calculate the mean of the RGB weights
        mean_rgb_weights = original_conv1.weight.mean(dim=1, keepdim=True)
        # Set the weights for the additional channels
        model.conv1.weight[:, 3:, :, :] = mean_rgb_weights.repeat(1, 3, 1, 1)
        # Scale all weights by 3/C
        model.conv1.weight.data *= 3 / NUM_CHANNELS  # Scaling factor for 6 channels


    # Remove the fully connected layer to get features instead of final classification output
    model = nn.Sequential(*list(model.children())[:-1])
    return model

class CombinedResNet(nn.Module):
    def __init__(self, model_name, num_classes):
        super(CombinedResNet, self).__init__()
        self.resnet1 = initialize_resnet(model_name)
        self.resnet2 = initialize_resnet(model_name)
        self.resnet3 = initialize_resnet(model_name)
        self.resnet4 = initialize_resnet(model_name)

        # Final fully connected layer
        if model_name == 'resnet50' or model_name == 'resnet34':
            num_in = 2048
        else:
            num_in = 512
            
        self.final_fc = nn.Linear(num_in*4, num_classes)

    def forward(self, x):
        # Split input into four parts and pass through the respective models
        x1 = self.resnet1(torch.cat((x[:, :3, :, :], x[:, 12:15, :, :]), dim=1)).squeeze()
        x2 = self.resnet2(torch.cat((x[:, 3:6, :, :], x[:, 15:18, :, :]), dim=1)).squeeze() 
        x3 = self.resnet3(torch.cat((x[:, 6:9, :, :], x[:, 18:21, :, :]), dim=1)).squeeze() 
        x4 = self.resnet4(torch.cat((x[:, 9:12, :, :], x[:, 21:24, :, :]), dim=1)).squeeze() 

        # Concatenate outputs
        combined = torch.cat((x1, x2, x3, x4), dim=1)

        # Pass through the final fully connected layer
        out = self.final_fc(combined)

        return out
