import numpy as np

def weighted_MAE(true, pred, num_bins = 6):
    true = np.array(true)
    pred = np.array(pred)

    min_label = 0.28095343708992004
    max_label = 3.094515085220337

    # Define bin edges
    bin_edges = np.linspace(min_label, max_label, num= num_bins+1)  # 6 bins -> 7 edges

    # Digitize the true labels into bins
    true_label_bins = np.digitize(true, bin_edges, right=True)

    # Initialize variables to store the weighted MAE
    weighted_mae = 0
    total_weight = 0

    # Calculate the MAE for each bin and weight it by the inverse of the bin's count
    for i in range(1, len(bin_edges)):  # Bin indices start at 1
        bin_mask = (true_label_bins == i)  # Mask to select items in the current bin
        
        # Ensure there are items in this bin
        if np.sum(bin_mask) > 0:
            current_labels = true[bin_mask]
            current_predictions = pred[bin_mask]
            bin_mae = np.mean(np.abs(current_labels - current_predictions))

            bin_count = len(current_labels)
            # Inverse weight is 1 / bin_count (avoid division by zero)
            bin_weight = 1 / bin_count if bin_count > 0 else 0
            weighted_mae += bin_mae * bin_weight
            total_weight += bin_weight

    # Normalize by the total weight to get the weighted average
    weighted_mae /= total_weight

    return weighted_mae