import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import sys

# Load the MNIST dataset (70,000 numbers)
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist["data"], mnist["target"]

# Convert data to numeric values
X = X.astype(np.float32)
y = y.astype(int)

# Split the dataset into training and testing sets
# 56,000 training, 14,000 testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to numpy arrays
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# Lower dimensionality for speedup
pca = PCA(n_components=50).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Normalize the pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Get training set
set_size = 10000
X_train_set = X_train[:set_size]
y_train_set = y_train[:set_size]


def knn_predict_fast(X_train, y_train, X_test, k=3):
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


# Number of neighbors to evaluate
default_k = 3
try:
    k = int(sys.argv[1])
except:
    k = default_k

# Predict and get accuracy
print(f"Data Loaded. Starting KNN with k={k}")
y_pred = knn_predict_fast(X_train_set, y_train_set, X_test, k)
print(f"Accuracy, {accuracy_score(y_test, y_pred)*100:.2f}%")
