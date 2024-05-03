import numpy as np

min_label = 0.28095343708992004
max_label = 3.094515085220337

def balanced_MAE(true, pred, num_bins=6):
    true = np.array(true)
    pred = np.array(pred)

    bin_edges = get_bin_edges(num_bins)
    true_label_bins = np.digitize(true, bin_edges, right=True)
    # Initialize variables to store the weighted MAE
    mae = 0 

    # Calculate the MAE for each bin and weight it by the inverse of the bin's count
    for i in range(0, num_bins):
        bin_mask = (true_label_bins == i)  # Mask to select items in the current bin
        
        current_labels = true[bin_mask]
        current_predictions = pred[bin_mask]
        mae += np.mean(np.abs(current_labels - current_predictions))

    return mae/num_bins

def get_bin_edges(num_bins=6):
    # Define bin edges
    bin_edges = np.linspace(min_label, max_label, num= num_bins+1)
    bin_edges = bin_edges[1:num_bins]
    return bin_edges