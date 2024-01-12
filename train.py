import torch
import torch.nn as nn
import torch.optim as optim
from checkpoint import save_checkpoint
from tqdm import tqdm
from dataloader import create_dataloader
from sklearn.metrics import f1_score

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
        total_train_samples = 0

        for idx, train_file in enumerate(train_files):
            train_loader = create_dataloader(train_file, batch_size)
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * inputs.size(0)
                total_train_samples += inputs.size(0)
        avg_train_loss = total_train_loss / total_train_samples

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for val_file in val_files:
                val_loader = create_dataloader(val_file, batch_size)
                for inputs, labels in tqdm(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item() * inputs.size(0)
                    total_val_samples += inputs.size(0)
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        # Calculate F1 scores
        f1_macro = f1_score(all_val_labels, all_val_preds, average='macro')
        f1_micro = f1_score(all_val_labels, all_val_preds, average='micro')
        avg_val_loss = total_val_loss / total_val_samples

        # Save checkpoint after each epoch
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            save_checkpoint(model, f'checkpoint_epoch_{epoch+1}.pth')

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, F1 Macro: {f1_macro:.4f}, F1 Micro: {f1_micro:.4f}')
    
    # Save the trained model
    save_checkpoint(model, f'checkpoint_epoch_{epoch+1}.pth')
    print('Training completed and model saved.')