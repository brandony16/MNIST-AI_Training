import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from cacheMNIST import load_mnist_cached


def load_and_preprocess_mnist(validation_split=0.2, one_hot=True):
    # Load MNIST dataset
    mnist = load_mnist_cached()
    data, labels = mnist["data"], mnist["target"]

    data = data.astype(np.float32)
    labels = labels.astype(int)

    # Normalize the data to [0, 1] range
    data = data / 255.0

    # One-hot encode the labels
    if one_hot:
        encoder = OneHotEncoder(sparse_output=False)
        labels = encoder.fit_transform(labels.reshape(-1, 1))

    # Split the data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=validation_split, random_state=42
    )

    # Convert to numpy arrays
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Num inputs in and num classifications
    input = 28 * 28
    output = 10

    return X_train, y_train, X_test, y_test, input, output, MNIST_LABELS

MNIST_LABELS = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}