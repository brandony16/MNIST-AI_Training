from numba import njit, prange
import numpy as np


@njit
def _gini_from_sqsum(sum_sq: float, total: int) -> float:
    """
    Compute Gini impurity from sum of squared counts:
        G = 1 - (sum_c n_c^2) / total^2
    """
    return 1.0 - sum_sq / (total * total)


@njit
def fast_best_split(
    data: np.ndarray, labels: np.ndarray, num_classes: int, max_features: int
):
    """
    Find the best feature index and threshold to split the data by minimizing
    weighted Gini impurity.

    Parameters
    ----------
    data : float32[npagesamples, n_features]
        Feature matrix at this node.
    labels : int32[n_samples]
        Class labels (0..n_classes–1).
    num_classes : int
        Number of unique classes.
    max_features : int
        Number of random features to consider (if >0), or 0 to use all.

    Returns
    -------
    best_feat : int or None
        Index of best feature, or None if no valid split found.
    best_thr : float or None
        Threshold for that feature (midpoint between adjacent values).
    """
    num_samples, num_features = data.shape
    if num_samples <= 1:
        return None, None

        # List of counts of each class in the current node
    parent_counts = np.bincount(labels, minlength=num_classes)

    parent_sum_sq = 0.0
    for c in range(num_classes):
        parent_sum_sq += parent_counts[c] * parent_counts[c]
    parent_gini = _gini_from_sqsum(parent_sum_sq, num_samples)

    if max_features and max_features < num_features:
        feature_indices = np.random.choice(num_features, max_features, replace=False)
    else:
        feature_indices = np.arange(num_features)

    # Calculate the gini impurity for the current node
    best_gini = parent_gini
    best_feature, best_threshold = None, None

    counts_left = np.zeros(num_classes, dtype=np.int64)
    counts_right = parent_counts.copy()

    for feature_idx in feature_indices:
        # Sort the samples by the current feature values.
        sorted_idx = np.argsort(data[:, feature_idx])
        feature_vals = data[sorted_idx, feature_idx]
        idx_labels = labels[sorted_idx]

        # Reset left/right counters
        counts_left[:] = 0
        counts_right[:] = parent_counts

        left_sum_sq = 0.0
        right_sum_sq = parent_sum_sq

        # Calculate cumulative sums for class counts
        for i in prange(1, num_samples):
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
            right_size = num_samples - i

            gini_left = _gini_from_sqsum(left_sum_sq, left_size)
            gini_right = _gini_from_sqsum(right_sum_sq, right_size)
            gini = (left_size * gini_left + right_size * gini_right) / num_samples

            # Update best gini and split point if current split is better
            if gini < best_gini:
                best_gini = gini
                best_feature = feature_idx
                best_threshold = (feature_vals[i] + feature_vals[i - 1]) / 2

    return best_feature, best_threshold
