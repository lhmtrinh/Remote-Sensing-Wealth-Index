import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """
    Trains the model.

    Args:
    - model (torch.nn.Module): The model to be trained.
    - train_loader (torch.utils.data.DataLoader): Training dataloader
    - val_loader (torch.utils.data.DataLoader): Validation dataloader
    - epochs (int): Number of training epochs.
    - learning_rate (float): Learning rate for the optimizer.
    """

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