import torch
import numpy as np
from tqdm import tqdm
def inference(model, checkpoint, loader, device):
    statedict = torch.load(checkpoint)
    model.load_state_dict(statedict)
    model.eval()
    model = model.to(device)

    # Collect all true labels and predictions
    true_labels = []
    predictions = []

    with torch.no_grad():
        for data, _,labels in tqdm(loader):
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data).squeeze()  # Assuming your model outputs a single value per sample
            true_labels.extend(labels.detach().cpu().numpy())
            predictions.extend(outputs.detach().cpu().numpy())
    
    return np.array(true_labels), np.array(predictions)