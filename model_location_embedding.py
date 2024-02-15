import torch
import torch.nn as nn
from torchvision.models import resnet50

NUM_CHANNELS = 24  # Number of input channels for your dataset
NUM_CLASSES = 1   # Number of output classes for your task
LOCATION_EMBEDDING = 256 # Embedding of location 

class ModifiedResNet50(nn.Module):
    def __init__(self):
        super(ModifiedResNet50, self).__init__()
        # Load a pre-trained ResNet-50 model
        original_model = resnet50(pretrained=True)
        
        # Modify the first convolutional layer to have NUM_CHANNELS input channels
        self.conv1 = nn.Conv2d(NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the weights explicitly for the RGB channels with pre-trained ImageNet weights
        with torch.no_grad():
            # First set of RGB channels
            self.conv1.weight[:, 0:3, :, :] = original_model.conv1.weight.clone()
            # Second set of RGB channels
            self.conv1.weight[:, 3:6, :, :] = original_model.conv1.weight.clone()
            # Third set of RGB channels
            self.conv1.weight[:, 6:9, :, :] = original_model.conv1.weight.clone()
            # Fourth set of RGB channels
            self.conv1.weight[:, 9:12, :, :] = original_model.conv1.weight.clone()

            # Initialize the weights for the non-RGB channels as the mean of the RGB channel weights
            mean_rgb_weights = original_model.conv1.weight.mean(dim=1, keepdim=True)
            self.conv1.weight[:, 12:, :, :] = mean_rgb_weights.repeat(1, NUM_CHANNELS - 12, 1, 1)

            # Scale all weights by 3/C
            self.conv1.weight.data *= 3 / NUM_CHANNELS

        # Use the remaining layers from the original ResNet-50 model
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool
        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4
        self.avgpool = original_model.avgpool
        
        # Modify the fully connected layer to match the desired number of output classes
        self.fc = nn.Linear(original_model.fc.in_features + LOCATION_EMBEDDING, NUM_CLASSES)
        
    def forward(self, x, location_embedding):
        # Define the forward pass, using the modified and original layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = torch.concat([x, location_embedding], dim = 1)
        x = self.fc(x)
        return x