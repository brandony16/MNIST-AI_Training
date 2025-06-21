import numpy as np
from numba import njit, prange
from typing import Tuple, Optional


@njit
def _gini_from_sqsum(sum_sq: float, total: int) -> float:
    """
    Compute Gini impurity from sum of squared counts:
        G = 1 - (sum_c n_c^2) / total^2
    """
    return 1.0 - sum_sq / (total * total)


@njit
def best_split(
    feature_matrix: np.ndarray, labels: np.ndarray, n_classes: int, max_features: int
) -> Tuple[Optional[int], Optional[float]]:
    """
    Find the best feature and threshold to split on by minimizing Gini.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (n_samples, n_features)
        Feature matrix for the current node.
    labels : np.ndarray, shape (n_samples,)
        Class labels (integers in [0, n_classes)).
    n_classes : int
        Number of distinct classes.
    max_features : int
        Number of random features to consider at this node,
        or 0 to consider all features.

    Returns
    -------
    best_feature : int or None
        Index of feature to split on, or None if no split found.
    best_threshold : float or None
        Threshold value (midpoint) for the split, or None.
    """

    # If no or one sample, no split is needed
    n_samples, n_features = feature_matrix.shape
    if n_samples <= 1:
        return None, None

    # List of counts of each class in the current node
    parent_counts = np.bincount(labels, minlength=n_classes)

    # Compute gini of parent
    parent_sum_sq = 0.0
    for c in range(n_classes):
        parent_sum_sq += parent_counts[c] * parent_counts[c]
    parent_gini = _gini_from_sqsum(parent_sum_sq, n_samples)

    if max_features and max_features < n_features:
        feature_indices = np.random.choice(n_features, max_features, replace=False)
    else:
        feature_indices = np.arange(n_features)

    # Calculate the gini impurity for the current node
    best_gini = parent_gini
    best_feature, best_threshold = None, None

    counts_left = np.zeros(n_classes, dtype=np.int64)
    counts_right = parent_counts.copy()

    for feature_idx in feature_indices:
        # Sort the samples by the current feature values.
        sorted_idx = np.argsort(feature_matrix[:, feature_idx])
        feature_vals = feature_matrix[sorted_idx, feature_idx]
        idx_labels = labels[sorted_idx]

        # Reset left/right counters
        counts_left[:] = 0
        counts_right[:] = parent_counts

        left_sum_sq = 0.0
        right_sum_sq = parent_sum_sq

        # Calculate cumulative sums for class counts
        for i in prange(1, n_samples):
            c = idx_labels[i - 1]
            old_left = counts_left[c]
            old_right = counts_right[c]
            counts_left[c] += 1
            counts_right[c] -= 1

            # Update sum of squares incrementally
            left_sum_sq += 2 * old_left + 1
            right_sum_sq += -2 * old_right + 1

            # Skip repeating thresholds
            if feature_vals[i] == feature_vals[i - 1]:
                continue

            # Calculate Gini impurities using cumulative sums
            left_size = i
            right_size = n_samples - i

            gini_left = _gini_from_sqsum(left_sum_sq, left_size)
            gini_right = _gini_from_sqsum(right_sum_sq, right_size)
            gini = (left_size * gini_left + right_size * gini_right) / n_samples

            # Update best gini and split point if current split is better
            if gini < best_gini:
                best_gini = gini
                best_feature = feature_idx
                best_threshold = (feature_vals[i] + feature_vals[i - 1]) / 2

    return best_feature, best_threshold
