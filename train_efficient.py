import torch
import torch.nn as nn
import torch.optim as optim
from checkpoint import save_checkpoint
from tqdm import tqdm
from dataloader import create_dataloader
from sklearn.metrics import f1_score
import threading
import queue

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

    def load_data(file, data_queue):
        loader = create_dataloader(file, False, batch_size)
        data_queue.put(loader)


    # Training and validation loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_samples = 0

        # Queue for DataLoader objects
        data_loader_queue = queue.Queue()
        data_loader_thread = None

        # Preload the first DataLoader
        if train_files:
            data_loader_thread = threading.Thread(target=load_data, args=(train_files[0], data_loader_queue))
            data_loader_thread.start()

        for idx, train_file in enumerate(train_files):
            # Wait for the DataLoader to be ready and retrieve it
            if data_loader_thread:
                data_loader_thread.join()
            train_loader = data_loader_queue.get()

            # Start loading the next DataLoader
            if idx + 1 < len(train_files):
                data_loader_thread = threading.Thread(target=load_data, args=(train_files[idx + 1], data_loader_queue))
                data_loader_thread.start()

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

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        all_val_labels = []
        all_val_preds = []
        val_loader_queue = queue.Queue()
        val_loader_thread = None

        if val_files:
            val_loader_thread = threading.Thread(target=load_data, args=(val_files[0], val_loader_queue))
            val_loader_thread.start()

        for idx, val_file in enumerate(val_files):
            if val_loader_thread:
                val_loader_thread.join()
            val_loader = val_loader_queue.get()

            if idx + 1 < len(val_files):
                val_loader_thread = threading.Thread(target=load_data, args=(val_files[idx + 1], val_loader_queue))
                val_loader_thread.start()

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item() * inputs.size(0)
                    total_val_samples += inputs.size(0)
                    all_val_labels.extend(labels.cpu().numpy())
                    all_val_preds.extend(outputs.argmax(dim=1).cpu().numpy())

        avg_val_loss = total_val_loss / total_val_samples
        f1_macro = f1_score(all_val_labels, all_val_preds, average='macro')
        f1_micro = f1_score(all_val_labels, all_val_preds, average='micro')

        # Print epoch summary
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, F1 Macro: {f1_macro:.4f}, F1 Micro: {f1_micro:.4f}')

        # Save model checkpoint
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            save_checkpoint(model, f'checkpoint_epoch_{epoch+1}.pth')

    # Save the trained model after all epochs
    save_checkpoint(model, 'final_model.pth')
    print('Training completed and final model saved.')