from KNN.KNNModel import knn_predict
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from DatasetFunctions.LoadData import load_and_preprocess_data
from Visualization import show_final_metrics
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a NN on MNIST or CIFAR"
    )
    parser.add_argument("--dataset", default="MNIST", help="Which dataset to load")
    parser.add_argument(
        "--valid-split", type=float, default=0.2, help="Validation split fraction"
    )
    parser.add_argument("--k", type=int, default=3, help="Number of neighbors")
    parser.add_argument(
        "--num-dims", type=int, default=50, help="Number of dimensions to reduce to"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    X_train, y_train, X_test, y_test, _, _, class_names = load_and_preprocess_data(
        validation_split=args.valid_split,
        one_hot=False,
        use_dataset=args.dataset,
        flatten=True,
    )

    # Lower dimensionality for speedup
    pca = PCA(n_components=args.num_dims).fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Get training set
    set_size = -1
    X_train_set = X_train[:set_size]
    y_train_set = y_train[:set_size]

    k = args.k

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
