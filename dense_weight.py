class DenseWeight:
    def __init__(self, alpha=0.5, epsilon=1e-4):
        self.alpha = alpha
        self.epsilon = epsilon
        self.min = None
        self.max = None
        self.mean_weight = None
        self.kde = None

    def fit(self, labels):
        # Create a gaussian Kernel Density Estimate
        self.kde = gaussian_kde(labels)

        # Evaluate the KDE for each point in labels
        pdf_values = self.kde(labels)
        self.min = np.min(pdf_values)
        self.max = np.max(pdf_values)

        # Calculate initial weights
        weights = self._weight_value(labels)

        # Set mean weight
        self.mean_weight = np.mean(weights)

    def _normalize_kde(self, labels):
        # Normalize KDE values
        normalized_pdf = (self.kde(labels) - self.min) / (self.max - self.min)
        return normalized_pdf

    def _weight_value(self, labels):
        # Calculate weight values
        normalized_pdf = self._normalize_kde(labels)
        weighted_values = np.maximum(1 - self.alpha * normalized_pdf, self.epsilon)
        return weighted_values

    def dense_weight(self, labels):
        # Apply dense weight normalization
        weights = self._weight_value(labels)
        dense_weights = weights / self.mean_weight
        return dense_weights