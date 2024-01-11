import torch
import torch.nn as nn
import torch.optim as optim
from checkpoint import save_checkpoint
from tqdm import tqdm
from dataloader import create_dataloader

def train_model(model, train_files, val_files, device, epochs=10, learning_rate=0.001, batch_size=64):
    """
    Trains the model and performs validation.

    Args:
    - model (torch.nn.Module): The model to be trained.
    - train_files (List[str]): List of training dataloaders.
    - val_files (List[str]): List of validation dataloaders.
    - epochs (int): Number of training epochs.
    - learning_rate (float): Learning rate for the optimizer.
    - batch_size (int): batch size
    """

    model = model.to(device)
    best_val_loss = 100000

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and validation loop
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for idx, train_file in enumerate(train_files):
            train_loader = create_dataloader(train_file, batch_size)
            print(f'Train data loader {idx+1}')
            for inputs, labels in tqdm(train_loader):

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            del(train_loader)

        avg_train_loss = total_train_loss / len(train_files)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for idx, val_file in enumerate(val_files):
                val_loader = create_dataloader(val_file, batch_size)
                print(f'Val data loader {idx+1}')
                for inputs, labels in tqdm(val_loader):

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                del(val_loader)

        avg_val_loss = total_val_loss / len(val_files)

        # Save checkpoint after each epoch
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            save_checkpoint(model, f'checkpoint_epoch_{epoch+1}.pth')

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Save the trained model
    save_checkpoint(model, f'checkpoint_epoch_{epoch+1}.pth')
    print('Training completed and model saved.')