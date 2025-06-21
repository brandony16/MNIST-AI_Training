from fastKNNModel import knn_predict_fast
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from cacheMNIST import load_mnist_cached
import numpy as np
import sys
import time


def main():
    # Load the MNIST dataset (70,000 numbers)
    mnist = load_mnist_cached()
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
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Lower dimensionality for speedup
    pca = PCA(n_components=50).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Normalize the pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Get training set
    set_size = 56000
    X_train_set = X_train[:set_size]
    y_train_set = y_train[:set_size]

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


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")
