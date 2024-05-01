import tensorflow as tf
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import h5py


BAND_KEYS = [
    'B4_1','B3_1','B2_1',
    'B4_2','B3_2','B2_2',
    'B4_3','B3_3','B2_3',
    'B4_4','B3_4','B2_4',
    'B8_1','B11_1','B12_1',
    'B8_2','B11_2','B12_2',
    'B8_3','B11_3','B12_3',
    'B8_4','B11_4','B12_4']

SCALAR_KEYS = ['latitude', 'longitude', 'wi']

LAND_COVER_KEY = ['label']

class SpectralImage:
    def __init__(self, bands_data, scalars):
        """
        Initializes a SpectralImage object.
        Args:
        - bands_data (dict): A dictionary with band names as keys and image tensors as values.
        - scalars (dict): A dictionary with scalar field names as keys and their values.
        """
        self.bands_data = bands_data
        self.scalars = scalars

    def get_band(self, band_name):
        """
        Returns the tensor for a specific band.
        Args:
        - band_name (str): The name of the band.
        Returns:
        - torch.Tensor: The tensor for the specified band.
        """
        return self.bands_data[band_name]

def read_tfrecord(tfrecord_path, size=225, with_landcover = False):
    """
    Reads a single TFRecord file and extracts the images and label.

    Args:
    - tfrecord_path (str): Path to the TFRecord file.
    - size (int): The size of each dimension for image data.

    Returns:
    - SpectralImage: The extracted spectral image object.
    - float: The extracted label.
    """
    raw_record = next(iter(tf.data.TFRecordDataset(tfrecord_path, compression_type="GZIP").take(1)))
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())

    # Extract scalar values
    scalars = {key: _extract_scalar_value(example, key) for key in SCALAR_KEYS}

    if with_landcover:
        # Extract image tensors
        images = {key: _extract_image_tensor(example, key, size) for key in BAND_KEYS + LAND_COVER_KEY}
    else:
        images = {key: _extract_image_tensor(example, key, size) for key in BAND_KEYS}

    # Create a SpectralImage object
    spectral_image = SpectralImage(images, scalars)

    # Extract the 'wi' label
    label = scalars.get('wi', 0)  # Default to 0 if 'wi' is not found
    
    return spectral_image_collate([(spectral_image, label)], with_landcover)

def _extract_scalar_value(example, key):
    # Extracts a scalar value from the TFRecord example
    values = example.features.feature[key].float_list.value
    return values[0] if values else None  # Return None or a default value if the list is empty

def _extract_image_tensor(example, key, size):
    # Extracts and converts an image tensor from the TFRecord example
    tensor = np.array(example.features.feature[key].float_list.value).reshape(size, size)
    return torch.tensor(tensor, dtype=torch.float)

def spectral_image_collate(batch, with_landcover = False):
    """
    Collate function to combine a list of SpectralImage objects and labels into a batch.

    Args:
    - batch (list): A list of tuples, each containing a SpectralImage object and a label.

    Returns:
    - Tuple containing two elements:
        - A tensor of shape [batch_size, channels, height, width] for the images.
        - A tensor of shape [batch_size] for the labels.
    """
    # Separate images and labels
    images, labels = zip(*batch)

    # Stack all channels of each SpectralImage object
    if with_landcover:
        batch_tensors = [torch.cat([image.bands_data[key].unsqueeze(0) for key in BAND_KEYS+LAND_COVER_KEY], dim=0) for image in images]
    else:
        batch_tensors = [torch.cat([image.bands_data[key].unsqueeze(0) for key in BAND_KEYS], dim=0) for image in images]

    # Stack all images to create a batch
    batch_tensor = torch.stack(batch_tensors, dim=0)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(labels, dtype=torch.float)

    return batch_tensor, labels_tensor

class ResizeTransform:
    """Transform to crop and resize the image to 224x224."""
    def __call__(self, x):
        return x[:, :224, :224]
    
class RescaleTransform:
    """Transform to scale image pixel values from 0-255 to 0-1."""
    def __call__(self, x):
        return x / 255.0
    
class NormalizeChannels:
    """
    Normalize each set of RGB channels independently.
    ImageNet mean and std are used for normalization.
    """
    def __init__(self):
        self.mean_RGB = [0.485, 0.456, 0.406]
        self.std_RGB = [0.229, 0.224, 0.225]
        # self.mean_nonRGB = [0.45,0.45,0.45]
        # self.std_nonRGB = [0.25,0.25,0.25]


    def __call__(self, x):
        x = x.clone()  # Clone to avoid modifying the original tensor

        # Normalize RGB channels
        mean_RGB = self.mean_RGB * 4
        std_RGB = self.mean_RGB*4        
        x[0:12] = transforms.functional.normalize(x[0:12], mean=mean_RGB, std=std_RGB)
        

        # Normalize Non RGB channels
        # mean_nonRGB = self.mean_nonRGB * 4
        # std_nonRGB = self.std_nonRGB * 4
        # x[12:24] = transforms.functional.normalize(x[12:24], mean=mean_nonRGB, std=std_nonRGB)

        return x

class ConcatenatedDataset(Dataset):
    def __init__(self, data_file, half):
        with h5py.File(data_file, 'r') as h5f:
            if half: 
                self.data = torch.from_numpy(h5f['data'][:]).float().half()
            else:
                self.data = torch.from_numpy(h5f['data'][:]).float()
            self.labels = torch.from_numpy(h5f['labels'][:]).float()
            self.locations = torch.from_numpy(h5f['locations'][:]).float()

        self.transform = transforms.Compose([
            ResizeTransform(),
            RescaleTransform(),
            NormalizeChannels()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        location = self.locations[idx]
        label = self.labels[idx]
        data = self.transform(data)
        return data, location, label
    
def create_dataset(file_paths, half=True):
    # Initialize lists to store datasets and labels
    datasets = []
    all_labels = []

    # Load each dataset and collect labels
    for file_path in file_paths:
        dataset = ConcatenatedDataset(file_path, half)
        datasets.append(dataset)
        labels = dataset.labels.numpy() 
        all_labels.extend(labels)
    
    # Combine all datasets
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # Store the combined labels as an attribute of the combined dataset
    combined_dataset.all_labels = torch.tensor(all_labels)
    
    return combined_dataset

def create_dataloader(dataset, batch_size, dense_weight_model=None, num_workers=0):
    if dense_weight_model != None:
        # Calculate weights for each data point in the combined dataset for downsampling        
        weights = dense_weight_model.dense_weight(dataset.all_labels)
        sampler = WeightedRandomSampler(weights, len(weights))
        # Create a DataLoader with the custom sampler
        return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler)
    
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)