import torch
import torch.nn as nn
import torch.optim as optim
from model import load_resnet_model  # Updated import
from dataloader import create_dataloader

def train_model(model, train_dir, val_dir, epochs=10, learning_rate=0.001):
    """
    Trains the model.

    Args:
    - model (torch.nn.Module): The model to be trained.
    - train_dir (str): Path to the training dataset.
    - val_dir (str): Path to the validation dataset.
    - epochs (int): Number of training epochs.
    - learning_rate (float): Learning rate for the optimizer.
    """
    # Load dataloaders
    train_loader, val_loader = create_dataloader(train_dir, val_dir)

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
# model = load_resnet_model('resnet18', 10)
# train_model(model, 'path/to/train', 'path/to/val')
