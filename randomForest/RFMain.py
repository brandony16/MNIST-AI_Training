from sklearn.metrics import accuracy_score
from RandomForest import RandomForest
from loadCIFAR import load_and_preprocess_cifar
from loadMNIST import load_and_preprocess_mnist
from Visualization import show_all_metrics
import pandas as pd
import time


def main():
    # Get relevant split and class name data
    X_train, y_train, X_test, y_test, _, _, class_names = load_and_preprocess_mnist(
        one_hot=False
    )

    sample_size = 10000
    X_train_sample = X_train[:sample_size]
    y_train_sample = y_train[:sample_size]

    # Initialize and train the RandomForestClassifier
    start = time.perf_counter()
    forest = RandomForest(num_trees=100, max_depth=15, max_features=56, n_jobs=4)
    forest.fit(X_train_sample, y_train_sample)

    # Make predictions and evaluate the model
    y_pred = forest.predict(X_test)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy, {accuracy * 100:.2f}%")

    history = []
    for i in range(10):
        history.append(
            {
                "epoch": i,
                "learning_rate": 0,
                "train_loss": None,
                "test_loss": None,
                "train_acc": None,
                "test_acc": accuracy,
            }
        )
    df = pd.DataFrame(history)

    show_all_metrics(y_test, y_pred, df, class_names)


if __name__ == "__main__":
    main()
