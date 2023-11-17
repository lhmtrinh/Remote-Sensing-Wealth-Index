import torch
from torchvision import transforms, datasets

def get_dataloaders(train_dir, val_dir, batch_size=32, img_size=224):
    """
    Creates dataloaders for the training and validation datasets.

    Args:
    - train_dir (str): Path to the training dataset directory.
    - val_dir (str): Path to the validation dataset directory.
    - batch_size (int): Batch size for dataloading.
    - img_size (int): Size to which the images will be resized.

    Returns:
    - train_loader, val_loader (tuple): Tuple of DataLoader instances for training and validation.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
