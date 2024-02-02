# Loaded dataloadder with half precision for faster finetuning
import torch
from checkpoint import save_checkpoint
from sklearn.metrics import r2_score
from torch.cuda.amp import autocast
from weighted_MAE import weighted_MAE

 
def train_model(model, criterion, optimizer, train_loaders, val_loaders, device, save_directory,epochs=10):
    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_samples = 0
        train_preds, train_labels = [], []
        
        for train_loader in train_loaders:
            for inputs, labels in train_loader:
                with autocast():
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item() * inputs.size(0)
                total_train_samples += inputs.size(0)
                train_preds.extend(outputs.detach().cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_train_loss / total_train_samples
        train_r2 = r2_score(train_labels, train_preds)
        train_mae = weighted_MAE(train_labels, train_preds)

        model.eval()
        total_val_loss = 0
        total_val_samples = 0
        val_preds, val_labels = [], []

        for val_loader in val_loaders:

            with torch.no_grad():
                for inputs, labels in val_loader:
                    with autocast():
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs).squeeze()
                        loss = criterion(outputs, labels)
                        total_val_loss += loss.item() * inputs.size(0)
                    total_val_samples += inputs.size(0)
                    val_preds.extend(outputs.detach().cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / total_val_samples
        val_r2 = r2_score(val_labels, val_preds)
        val_mae = weighted_MAE(val_labels, val_preds)

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train R2: {train_r2:.4f}, Train weighted MAE: {train_mae:.4f},Val Loss: {avg_val_loss:.4f}, Val R2: {val_r2:.4f}, Val weighted MAE: {val_mae:.4f}')

        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            save_checkpoint(model, f'{save_directory}/checkpoint_epoch_{epoch+1}.pth')

    save_checkpoint(model, f'{save_directory}/final_model.pth')
    print('Training completed and final model saved.')
