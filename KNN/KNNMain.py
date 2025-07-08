from KNN.KNNModel import knn_predict
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from DatasetFunctions.LoadData import load_and_preprocess_data
from Visualization import show_final_metrics
import sys
import time


def main():

    X_train, y_train, X_test, y_test, _, _, class_names = load_and_preprocess_data(
        one_hot=False, use_dataset="MNIST"
    )

    # Lower dimensionality for speedup
    pca = PCA(n_components=50).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Get training set
    set_size = -1
    X_train_set = X_train[:set_size]
    y_train_set = y_train[:set_size]

    # Number of neighbors to evaluate
    default_k = 3
    try:
        k = int(sys.argv[1])
    except:
        k = default_k

    # Predict and get metrics
    start = time.perf_counter()
    print(f"Data Loaded. Starting KNN with k={k}")
    y_pred = knn_predict(X_train_set, y_train_set, X_test, k)
    accuracy = accuracy_score(y_test, y_pred)

    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")

    print(f"Accuracy, {accuracy * 100:.2f}%")

    show_final_metrics(y_test, y_pred, class_names)


if __name__ == "__main__":
    main()
