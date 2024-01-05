import tensorflow as tf
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

BAND_KEYS = ['B2_1','B2_2','B2_3','B2_4','B3_1','B3_2','B3_3','B3_4','B4_1','B4_2','B4_3','B4_4','B8_1','B8_2','B8_3','B8_4','B11_1','B11_2','B11_3','B11_4','B12_1','B12_2','B12_3','B12_4']
SCALAR_KEYS = ['latitude', 'longitude', 'wi']
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

    def plot(self, band_keys, min_val=0.0, max_val=0.3):
        """
        Plots an image using specified bands as RGB channels. 
        Args:
        - band_keys (list): The keys for the bands to be used as RGB channels. E.g. ['B4_4', 'B3_4', 'B2_4']
        - min_val (float): Minimum value for normalization.
        - max_val (float): Maximum value for normalization.
        """
        # Initialize an empty array for the image
        arr = np.zeros((self.bands_data[band_keys[0]].shape[0], self.bands_data[band_keys[0]].shape[1], 3))

        # Extract and normalize the bands
        for i, key in enumerate(band_keys):
            band_data = self.bands_data[key].numpy()
            arr[:, :, i] = self._normalize_and_clip(band_data, min_val, max_val)

        # Plot the image
        plt.imshow(arr)
        plt.axis('off')
        plt.show()

    def _normalize_and_clip(self, band_data, min_val, max_val):
        """
        Normalizes and clips the band data.
        Args:
        - band_data (numpy.ndarray): Band data to be normalized and clipped.
        - min_val (float): Minimum value for normalization.
        - max_val (float): Maximum value for normalization.
        Returns:
        - numpy.ndarray: Normalized and clipped band data.
        """
        return np.clip(band_data, min_val, max_val) / (max_val - min_val)
    
    ### TODO: Implement code to peform transformation for RGB channels. Figure out the statistics for other channels

class TFRecordImageDataset(Dataset):
    def __init__(self, file_paths, size=225):
        """
        Custom dataset for reading TFRecord files.
        Args:
        - file_paths (list): List of paths to TFRecord files.
        - image_keys (list): List of keys for image data in the TFRecord files.
        - scalar_keys (list): List of keys for scalar data in the TFRecord files.
        - size (int): The size of each dimension for image data.
        """
        self.file_paths = file_paths
        self.size = size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        raw_record = next(iter(tf.data.TFRecordDataset(self.file_paths[idx], compression_type="GZIP").take(1)))
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())

        # Extract scalar values
        scalars = {key: self._extract_scalar_value(example, key) for key in SCALAR_KEYS}

        # Extract image tensors
        images = {key: self._extract_image_tensor(example, key) for key in BAND_KEYS}

        # Create a SpectralImage object
        spectral_image = SpectralImage(images, scalars)
        return spectral_image

    def _extract_scalar_value(self, example, key):
        # Extracts a scalar value from the TFRecord example
        values = example.features.feature[key].float_list.value
        return values[0] if values else None  # Return None or a default value if the list is empty

    def _extract_image_tensor(self, example, key):
        # Extracts and converts an image tensor from the TFRecord example
        tensor = np.array(example.features.feature[key].float_list.value).reshape(self.size, self.size)
        return torch.tensor(tensor, dtype=torch.float)
    

def spectral_image_collate(batch):
    """
    Collate function to combine a list of SpectralImage objects into a batch tensor.

    Args:
    - batch (list): A list of SpectralImage objects.

    Returns:
    - torch.Tensor: A tensor of shape [batch_size, channels, height, width]
    """
    # Ensure each tensor is of shape [1, height, width]
    batch_tensors = [torch.cat([image.bands_data[key].unsqueeze(0) for key in BAND_KEYS], dim=0) for image in batch]

    # Stack all images to create a batch
    batch_tensor = torch.stack(batch_tensors, dim=0)

    return batch_tensor


def create_dataloader(file_paths, batch_size, num_workers=4, shuffle=True):
    dataset = TFRecordImageDataset(file_paths)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=spectral_image_collate)
    return data_loader
