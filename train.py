import torch
import torch.nn as nn
import torch.optim as optim
from model import load_model
from dataloader import get_dataloaders

def train_model(model_name, num_classes, train_dir, val_dir, epochs=10, learning_rate=0.001):
    """
    Trains the model.

    Args:
    - model_name (str): ResNet model to use.
    - num_classes (int): Number of output classes.
    - train_dir (str): Path to the training dataset.
    - val_dir (str): Path to the validation dataset.
    - epochs (int): Number of training epochs.
    - learning_rate (float): Learning rate for the optimizer.
    """
    # Load model and dataloaders
    model = load_model(model_name, num_classes)
    train_loader, val_loader = get_dataloaders(train_dir, val_dir)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs} completed.')

    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')
    print('Training completed and model saved.')

# Example usage
# train_model('resnet18', 10, 'path/to/train', 'path/to/val')
