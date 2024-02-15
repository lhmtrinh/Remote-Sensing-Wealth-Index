import re
import os
from matplotlib import pyplot as plt

def get_file_paths(directory):
    """
    Get all file paths in the specified directory with the specified file extension.

    Args:
    - directory (str): The directory to search for files.
    - file_extension (str): The file extension to filter by.

    Returns:
    - List[str]: A list of file paths.
    """
    file_paths = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        file_paths.append(file_path)
    return file_paths

# Use a regular expression to extract the batch number from the filename
def extract_batch_number(file_path):
    match = re.search(r"data_batch_(\d+)", file_path)
    if match:
        return int(match.group(1))
    else:
        return -1  # If for some reason a file doesn't match the pattern
    
def sorted_file_paths(directory):
    return sorted(get_file_paths(directory), key=extract_batch_number)

def plot_scatter(true_labels, predictions, experiment, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.title(f'{experiment} true vs. prediction')
    plt.scatter(true_labels, predictions, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.plot([true_labels.min(), true_labels.max()], [true_labels.min(), true_labels.max()], 'k--', lw=4)
        # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(f'{save_path}/scatter', bbox_inches='tight')
    
    plt.show()



def plot_residual(true_labels, predictions, experiment, save_path=None):
    # Calculate residuals
    residuals = [true - pred for true, pred in zip(true_labels, predictions)]

    # Plotting the residual plot
    plt.figure(figsize=(6, 6))
    plt.scatter(true_labels, residuals, color='blue', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='--')  # Adds a horizontal line at zero for reference
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.title(f'{experiment} Residual Plot')
    # Setting axis limits
    plt.xlim([min(true_labels), max(true_labels)])  # Set x-axis limits to min and max of true values
    plt.ylim([-2, 2])  # Set y-axis limits to -2 and 2 for residuals
        # Save the plot if a save path is provided
    if save_path is not None:
        plt.savefig(f'{save_path}/residual', bbox_inches='tight')
        
    plt.show()


