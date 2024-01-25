import torch
import torch.nn as nn
import torch.optim as optim
from checkpoint import save_checkpoint
from tqdm import tqdm
from dataloader import create_dataloader
import threading
import queue
from sklearn.metrics import r2_score
from weighted_MSE_loss import WeightedMSELoss


def train_model(model, train_files, val_files, device, dense_weight_model,epochs=10, learning_rate=0.001, batch_size=64):
    model = model.to(device)
    best_val_loss = float('inf')

    criterion = WeightedMSELoss(dense_weight_model)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def load_data(file, data_queue):
        loader = create_dataloader(file, True, batch_size)
        data_queue.put(loader)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_samples = 0
        train_preds, train_labels = [], []

        data_loader_queue = queue.Queue()
        data_loader_thread = threading.Thread(target=load_data, args=(train_files[0], data_loader_queue))
        data_loader_thread.start()

        for idx, train_file in enumerate(train_files):
            data_loader_thread.join()
            train_loader = data_loader_queue.get()

            if idx + 1 < len(train_files):
                data_loader_thread = threading.Thread(target=load_data, args=(train_files[idx + 1], data_loader_queue))
                data_loader_thread.start()

            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels, dense_weight_model)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * inputs.size(0)
                total_train_samples += inputs.size(0)
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_train_loss / total_train_samples
        train_r2 = r2_score(train_labels, train_preds)

        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        val_preds, val_labels = [], []

        val_loader_queue = queue.Queue()
        val_loader_thread = threading.Thread(target=load_data, args=(val_files[0], val_loader_queue))
        val_loader_thread.start()

        for idx, val_file in enumerate(val_files):
            val_loader_thread.join()
            val_loader = val_loader_queue.get()

            if idx + 1 < len(val_files):
                val_loader_thread = threading.Thread(target=load_data, args=(val_files[idx + 1], val_loader_queue))
                val_loader_thread.start()

            with torch.no_grad():
                for inputs, labels in tqdm(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels, dense_weight_model)
                    total_val_loss += loss.item() * inputs.size(0)
                    total_val_samples += inputs.size(0)
                    val_preds.extend(outputs.detach().cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / total_val_samples
        val_r2 = r2_score(val_labels, val_preds)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train R2: {train_r2:.4f}, Val Loss: {avg_val_loss:.4f}, Val R2: {val_r2:.4f}')

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            save_checkpoint(model, f'checkpoint_epoch_{epoch+1}.pth')

    save_checkpoint(model, 'final_model.pth')
    print('Training completed and final model saved.')
