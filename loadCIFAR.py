import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from cacheCIFAR import load_cifar_cached


def load_and_preprocess_cifar(validation_split=0.2):
    # Load CIFAR-10 dataset
    cifar = load_cifar_cached()
    data, labels = cifar["data"], cifar["target"]

    data = data.astype(np.float32)
    labels = labels.astype(int)

    # Normalize the data to [0, 1] range
    data = data / 255.0

    # One-hot encode the labels
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
    input = 32 * 32 * 3
    output = 10
    return X_train, y_train, X_test, y_test, input, output
