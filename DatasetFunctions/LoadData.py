import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from DatasetFunctions.CacheMNIST import load_mnist_cached
from DatasetFunctions.CacheCIFAR import load_cifar_cached
from DatasetFunctions.ClassNames import CLASS_NAMES_BY_DATASET


def load_and_preprocess_data(validation_split=0.2, one_hot=True, use_dataset="MNIST"):
    # Load dataset
    match (use_dataset):
        case "MNIST":
            dataset = load_mnist_cached()
        case "CIFAR":
            dataset = load_cifar_cached()
        case _:
            dataset = load_mnist_cached()

    data, labels = dataset["data"], dataset["target"]

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
    input = X_train.shape[1]

    if one_hot:
        output = len(set(np.argmax(y_test, axis=1)))
    else:
        output = len(set(y_test))

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        input,
        output,
        CLASS_NAMES_BY_DATASET[use_dataset],
    )


def load_cnn_data(validation_split=0.2, one_hot=True, use_dataset="MNIST"):
    # Load dataset
    match (use_dataset):
        case "MNIST":
            dataset = load_mnist_cached()
            c, h, w = 1, 28, 28
            MEAN = np.array([0.1307], dtype=np.float32)[:, None, None]
            STD = np.array([0.3081], dtype=np.float32)[:, None, None]
        case "CIFAR":
            dataset = load_cifar_cached()
            c, h, w = 3, 32, 32
            MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)[:, None, None]
            STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)[:, None, None]
        case _:
            dataset = load_mnist_cached()
            c, h, w = 1, 28, 28

    data, labels = dataset["data"], dataset["target"]

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

    # Reshape into appropriate sizes
    X_train = X_train.reshape(-1, c, h, w)
    X_test = X_test.reshape(-1, c, h, w)

    X_train = (X_train - MEAN) / STD
    X_test = (X_test - MEAN) / STD

    # Num inputs in and num classifications
    input = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
    output = len(set(np.argmax(y_test, axis=1)))

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        input,
        output,
        CLASS_NAMES_BY_DATASET[use_dataset],
    )
