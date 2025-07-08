from sklearn.metrics import accuracy_score
from RandomForest import RandomForest
from DatasetFunctions.LoadData import load_and_preprocess_data
from Visualization import show_final_metrics
import time


def main():
    print("Starting RF Progam")
    # Get relevant split and class name data
    X_train, y_train, X_test, y_test, _, _, class_names = load_and_preprocess_data(
        one_hot=False, use_dataset="CIFAR"
    )

    sample_size = -1
    X_train_sample = X_train[:sample_size]
    y_train_sample = y_train[:sample_size]

    # Initialize and train the RandomForestClassifier
    start = time.perf_counter()
    forest = RandomForest(num_trees=300, max_depth=25, max_features=64, n_jobs=8)

    print("Building Forest")
    forest.fit(X_train_sample, y_train_sample)

    # Make predictions and evaluate the model
    y_pred = forest.predict(X_test)
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy, {accuracy * 100:.2f}%")

    show_final_metrics(y_test, y_pred, class_names)


if __name__ == "__main__":
    main()
