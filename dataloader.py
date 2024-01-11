import tensorflow as tf
import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms


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

def spectral_image_collate(batch):
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
    batch_tensors = [torch.cat([image.bands_data[key].unsqueeze(0) for key in BAND_KEYS], dim=0) for image in images]

    # Stack all images to create a batch
    batch_tensor = torch.stack(batch_tensors, dim=0)

    # Convert labels to a tensor
    labels_tensor = torch.tensor(labels, dtype=torch.float)

    return batch_tensor, labels_tensor

def read_tfrecord(tfrecord_path, size=225):
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

    # Extract image tensors
    images = {key: _extract_image_tensor(example, key, size) for key in BAND_KEYS}

    # Create a SpectralImage object
    spectral_image = SpectralImage(images, scalars)

    # Extract the 'wi' label
    label = scalars.get('wi', 0)  # Default to 0 if 'wi' is not found
    
    return spectral_image_collate([(spectral_image, label)])

def _extract_scalar_value(example, key):
    # Extracts a scalar value from the TFRecord example
    values = example.features.feature[key].float_list.value
    return values[0] if values else None  # Return None or a default value if the list is empty

def _extract_image_tensor(example, key, size):
    # Extracts and converts an image tensor from the TFRecord example
    tensor = np.array(example.features.feature[key].float_list.value).reshape(size, size)
    return torch.tensor(tensor, dtype=torch.float)

class ResizeTransform:
    """Transform to crop and resize the image to 224x224."""
    def __call__(self, x):
        return x[:, :224, :224]

class ConcatenatedDataset(Dataset):
    def __init__(self, data_file):
        self.data = torch.load(data_file)['data']
        self.labels = torch.load(data_file)['labels']
        self.transform = transforms.Compose([
            ResizeTransform()
        ])
        # Define bin edges for label binning
        min_label = 0.280953
        max_label = 3.171482
        self.bin_edges = np.linspace(min_label, max_label, num=11)  # 10 bins -> 11 edges

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data, label = self.data[idx], self.labels[idx]
        if self.transform:
            data = self.transform(data)

        # Bin the label
        binned_label = np.digitize(label, self.bin_edges) - 1  # Subtract 1 to get bins from 0 to 9

        return data, binned_label
    

def create_dataloader(file_path, batch_size, num_workers=4):
    dataset = ConcatenatedDataset(file_path)
    # Shuffle to false because when we create pth files for data, we have already shuffled them
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers= num_workers, shuffle=False)
    return data_loader