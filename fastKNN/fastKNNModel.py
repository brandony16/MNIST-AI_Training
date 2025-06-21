import numpy as np
from numba import njit, prange, float32


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def knn_predict_fast(data_train, labels_train, data_test, k=3, num_classes=10):
    """
    Perform k-Nearest Neighbors classification in a fully vectorized way.

    This function computes the squared Euclidean distance between each test
    point and all training points using the identity:

        ||t_i - x_j||^2 = ||t_i||^2 + ||x_j||^2 - 2 * t_i · x_j

    It then finds the indices of the k smallest distances for each test
    example, retrieves their labels, and returns a majority-vote prediction.

    Parameters
    ----------
    data_train : np.ndarray, shape (n_train, n_features)
        Training data matrix where each row is a feature vector.
    labels_train : np.ndarray, shape (n_train,)
        Integer labels for the training data.
    data_test : np.ndarray, shape (n_test, n_features)
        Test data matrix where each row is a feature vector to classify.
    k : int, default=3
        Number of nearest neighbors to use for voting.
    num_classes : int, default=10
        Number of classes the data falls into

    Returns
    -------
    predictions : np.ndarray, shape (n_test,)
        Predicted class labels for each test point. Each entry is the
        mode of its k nearest neighbors in the training set.
    """
    n_train, n_features = data_train.shape
    n_test = data_test.shape[0]
    predictions = np.empty(n_test, dtype=np.int32)

    # Compute squared norms of training data
    train_norms = np.empty(n_train, dtype=np.float32)
    for train_idx in range(n_train):
        sum = float32(0.0)
        for feat_idx in range(n_features):
            sum += data_train[train_idx, feat_idx] * data_train[train_idx, feat_idx]
        train_norms[train_idx] = sum

    for test_idx in prange(n_test):
        # Compute norm vector for this test point
        test_vector = data_test[test_idx]
        test_norm = float32(0.0)
        for feat_idx in range(n_features):
            test_norm += test_vector[feat_idx] * test_vector[feat_idx]

        # Compute distances to each training point
        distances = np.empty(n_train, dtype=np.float32)
        for train_idx in range(n_train):
            dot = float32(0.0)
            for feat_idx in range(n_features):
                dot += test_vector[feat_idx] * data_train[train_idx, feat_idx]
            distances[train_idx] = test_norm + train_norms[train_idx] - 2 * dot

        # Do majority vote on closest k neighbors
        indexes = np.argsort(distances)
        counts = np.zeros(num_classes, dtype=np.int32)
        for neighbor in range(k):
            label = labels_train[indexes[neighbor]]
            counts[label] += 1

        # Find highest value
        best = 0
        best_count = counts[0]
        for c in range(1, num_classes):
            if counts[c] > best_count:
                best = c
                best_count = counts[c]

        predictions[test_idx] = best

    return predictions
