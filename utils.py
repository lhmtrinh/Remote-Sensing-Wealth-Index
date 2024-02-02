import re
import os

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

