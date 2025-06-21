import numpy as np


def knn_predict(X_train, y_train, X_test, k=3):
    """
    Perform k-Nearest Neighbors classification in a fully vectorized way.

    This function computes the squared Euclidean distance between each test
    point and all training points using the identity:

        ||t_i - x_j||^2 = ||t_i||^2 + ||x_j||^2 - 2 * t_i · x_j

    It then finds the indices of the k smallest distances for each test
    example, retrieves their labels, and returns a majority-vote prediction.

    Parameters
    ----------
    X_train : np.ndarray, shape (n_train, n_features)
        Training data matrix where each row is a feature vector.
    y_train : np.ndarray, shape (n_train,)
        Integer labels for the training data.
    X_test : np.ndarray, shape (n_test, n_features)
        Test data matrix where each row is a feature vector to classify.
    k : int, default=3
        Number of nearest neighbors to use for voting.

    Returns
    -------
    y_pred : np.ndarray, shape (n_test,)
        Predicted class labels for each test point. Each entry is the
        mode of its k nearest neighbors in the training set.
    """
    # Compute squared norms to allow for matrix ops
    train_norms = np.sum(X_train**2, axis=1)
    test_norms = np.sum(X_test**2, axis=1)

    # Build the matrix of squares distances
    # dists2[i, j] = ||t_i - x_j||^2 (Euclidean distance)
    dists_squared = (
        test_norms[:, None] + train_norms[None, :] - 2.0 * X_test.dot(X_train.T)
    )

    # Get indicies and labels
    knn_indices = np.argpartition(dists_squared, k, axis=1)[:, :k]
    knn_labels = y_train[knn_indices]

    # Majority vote
    y_pred = np.empty(knn_labels.shape[0], dtype=int)
    for i, neighbors in enumerate(knn_labels):
        y_pred[i] = np.bincount(neighbors, minlength=10).argmax()

    return y_pred
