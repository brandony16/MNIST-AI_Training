from fastKNNModel import knn_predict_fast
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from loadCIFAR import load_and_preprocess_cifar
from loadMNIST import load_and_preprocess_mnist
import pandas as pd
from Visualization import show_all_metrics
import numpy as np
import sys
import time


def main():
    start = time.perf_counter()

    X_train, y_train, X_test, y_test, _, _ = load_and_preprocess_mnist()

    # Lower dimensionality for speedup
    pca = PCA(n_components=50).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Undo one hot encoding
    y_test = np.argmax(y_test, axis=1).astype(np.int32)
    y_train = np.argmax(y_train, axis=1).astype(np.int32)

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

    # Predict and get metrics
    print(f"Data Loaded. Starting KNN with k={k}")
    y_pred = knn_predict_fast(X_train_set, y_train_set, X_test, k)
    accuracy = accuracy_score(y_test, y_pred)

    history = []
    for i in range(10):
        history.append(
            {
                "epoch": i,
                "learning_rate": 0,
                "train_loss": 0,
                "test_loss": 0,
                "train_acc": accuracy,
                "test_acc": accuracy,
            }
        )
    df = pd.DataFrame(history)

    print(f"Accuracy, {accuracy * 100:.2f}%")
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")

    show_all_metrics(y_test, y_pred, df)


if __name__ == "__main__":
    main()
