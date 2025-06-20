from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from RandomForest import RandomForest
from cacheMNIST import load_mnist_cached
import numpy as np
import time


def main():
    # Step 1: Load the MNIST dataset
    mnist = load_mnist_cached()
    X, y = mnist["data"], mnist["target"]

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Step 3: Normalize pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Step 4: Convert labels to integers
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    sample_size = 1000
    X_train_sample = X_train[:sample_size]
    y_train_sample = y_train[:sample_size]

    # Step 5: Initialize and train the RandomForestClassifier
    forest = RandomForest(n_trees=100, max_depth=15, n_jobs=-1, max_features=56)
    forest.fit(X_train_sample, y_train_sample)

    # Step 6: Make predictions and evaluate the model
    y_pred = forest.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)

    # Print the results
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")
