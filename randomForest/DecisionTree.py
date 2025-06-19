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

    n_samples, n_features = feature_matrix.shape
    if n_samples <= 1:
        return None, None

    # List of counts of each class in the current node
    parent_counts = np.bincount(labels, minlength=n_classes)

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


class DecisionTree:
    def __init__(
        self,
        max_depth: int = None,
        max_features: int = None,
        min_samples_split: int = 2,
    ):
        self.max_depth = max_depth or np.inf
        self.max_features = max_features
        self.min_samples_split = min_samples_split

    # X is the data, y is the labels
    # Fits the tree to the data
    def fit(self, X, y):
        # Number of classes. 10 bc 0-9
        self.n_classes_ = len(set(y))

        # Number of features.
        self.n_features_ = X.shape[1]

        self.tree_ = self._grow_tree(X, y, depth=0)

    def predict(self, X):
        # Calls predict for each input and returns predicted labels
        return np.array([self._predict(inputs) for inputs in X], dtype=int)

    def _best_split(self, X, y):
        # Find the best split using the optimized best_split function
        return best_split(X, y, self.n_classes_, self.max_features)

    # Recursively builds the decision tree
    def _grow_tree(self, X, y, depth=0):
        # Terminal if pure or too small or too deep
        if (
            len(y) < self.min_samples_split
            or depth >= self.max_depth
            or np.all(y == y[0])
        ):
            counts = np.bincount(y, minlength=self.n_classes_)
            pred = int(np.argmax(counts))
            gini = 1.0 - np.sum((counts / len(y)) ** 2)
            return Node(gini, len(y), counts, pred)

            # Parent counts & impurity
        parent_counts = np.bincount(y, minlength=self.n_classes_)
        parent_gini = 1.0 - np.sum((parent_counts / len(y)) ** 2)
        idx, thr = self._best_split(X, y)

        # If no split found, make leaf
        if idx is None:
            pred = int(np.argmax(parent_counts))
            return Node(parent_gini, len(y), parent_counts, pred)

        # Make new node and recursively grow tree
        mask_left = X[:, idx] < thr
        X_left, y_left = X[mask_left], y[mask_left]
        X_right, y_right = X[~mask_left], y[~mask_left]

        node = Node(parent_gini, len(y), parent_counts, int(np.argmax(parent_counts)))
        node.feature_index = idx
        node.threshold = thr

        node.left = self._grow_tree(X_left, y_left, depth + 1)
        node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while node.left is not None:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


class Node:
    __slots__ = (
        "gini",
        "num_samples",
        "num_samples_per_class",
        "predicted_class",
        "feature_index",
        "threshold",
        "left",
        "right",
    )

    def __init__(
        self,
        gini: float,
        num_samples: int,
        num_samples_per_class: np.ndarray,
        predicted_class: int,
    ):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
