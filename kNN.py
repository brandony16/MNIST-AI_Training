import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter

# Load the MNIST dataset
mnist = fetch_openml("mnist_784", version=1)
X, y = mnist["data"], mnist["target"]

# Convert data to numeric values
X = X.astype(np.float32)
y = y.astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize the pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

train_size = 1000  # Set the number of samples you want in your smaller training set
X_train_small = X_train[:train_size]
y_train_small = y_train[:train_size]


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def knn_predict(X_train, y_train, X_test, k=3):
    """
    Perform k-Nearest Neighbors classification.

    This function computes the Euclidean distance between each test
    point and all training points.

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
    y_pred = []
    node_counter = 0

    for _, test_point in enumerate(X_test.values):
        node_counter = node_counter + 1
        # Compute distances from test_point to all training points
        distances = [
            euclidean_distance(test_point, x_train) for x_train in X_train.values
        ]

        # Get the k nearest neighbors
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = [y_train.iloc[i] for i in k_indices]

        # Determine the most common class label among the neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        y_pred.append(most_common[0][0])

    return np.array(y_pred)


# Predict the labels for the test set
k = 3
y_pred = knn_predict(X_train_small, y_train_small, X_test, k)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
