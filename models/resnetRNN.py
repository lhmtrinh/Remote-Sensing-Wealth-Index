import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights

NUM_CHANNELS = 24

class ResNetRNN(nn.Module):
    def __init__(self, num_classes=1, rnn_hidden_size=128, rnn_num_layers=1, bidirectional=False):
        super(ResNetRNN, self).__init__()
        
        # Load and modify ResNet
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()  # Remove the fully connected layer
        self.resnet = modify_resnet_model(self.resnet)
        
        # RNN parameters
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.bidirectional = bidirectional
        rnn_directions = 2 if bidirectional else 1
        
        # Define RNN layer
        self.rnn = nn.LSTM(input_size=self.resnet.fc.in_features, hidden_size=rnn_hidden_size, 
                           num_layers=rnn_num_layers, batch_first=True, bidirectional=bidirectional)
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden_size * rnn_directions, num_classes)
        
    def forward(self, x):
        batch_size, C, seq_len, H, W = x.size()
        # Reshape input for ResNet: (batch_size * seq_len, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)
        
        # Extract features with ResNet
        with torch.no_grad():
            features = self.resnet(x)
        
        # Reshape features for RNN: (batch_size, seq_len, feature_size)
        features = features.view(batch_size, seq_len, -1)
        
        # RNN forward pass
        rnn_out, _ = self.rnn(features)
        
        # Use the output of the last RNN cell
        if self.bidirectional:
            rnn_out = rnn_out[:, -1, :self.rnn_hidden_size] + rnn_out[:, -1, self.rnn_hidden_size:]
        else:
            rnn_out = rnn_out[:, -1, :]
        
        # Fully connected layer
        output = self.fc(rnn_out)
        
        return output

def modify_resnet_model(model):
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