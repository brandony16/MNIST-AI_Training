import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_and_preprocess_mnist(validation_split=0.2):
    # Load MNIST dataset
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    data, labels = mnist["data"], mnist["target"]

    data = data.astype(np.float32)
    labels = labels.astype(int)

    # Normalize the data to [0, 1] range
    data = data / 255.0

    # One-hot encode the labels
    encoder = OneHotEncoder(sparse_output=False)
    labels = encoder.fit_transform(labels.reshape(-1, 1))

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=validation_split, random_state=42
    )

    # Convert to numpy arrays
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, y_train, X_val, y_val
