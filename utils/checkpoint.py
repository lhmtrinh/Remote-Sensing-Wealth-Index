import torch

def save_checkpoint(model, filename='model_checkpoint.pth'):
    """
    Saves the model checkpoint.

    Args:
    - model (torch.nn.Module): The model to save.
    - filename (str): The filename for the saved checkpoint.
    """
    torch.save(model.state_dict(), filename)
    print(f'Model saved as {filename}')

def load_checkpoint(model, filename='model_checkpoint.pth'):
    """
    Loads the model checkpoint.

    Args:
    - model (torch.nn.Module): The model to load the checkpoint into.
    - filename (str): The filename of the checkpoint to load.

    Returns:
    - model (torch.nn.Module): The model with the checkpoint loaded.
    """
    model.load_state_dict(torch.load(filename))
    model.eval()
    print(f'Loaded checkpoint from {filename}')
    return model
