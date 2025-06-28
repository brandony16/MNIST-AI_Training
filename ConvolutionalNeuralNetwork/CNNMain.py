import numpy as np
import cupy as cp
import pandas as pd
from ConvolutionalNeuralNetwork.Layers.Conv2DLayer import Conv2D
from ConvolutionalNeuralNetwork.Layers.BatchNormLayer import BatchNorm2D
from ConvolutionalNeuralNetwork.Layers.ActivationLayers import ReLU
from ConvolutionalNeuralNetwork.Layers.DenseLayer import Dense
from ConvolutionalNeuralNetwork.Layers.DropoutLayer import Dropout
from ConvolutionalNeuralNetwork.Layers.FlattenLayer import Flatten
from ConvolutionalNeuralNetwork.Layers.PoolingLayer import MaxPool2D
from ConvolutionalNeuralNetwork.Layers.SoftmaxCELayer import SoftmaxCrossEntropy
from DatasetFunctions.LoadData import load_and_preprocess_data
from Sequential import Sequential
from SGD import use_optimizer
from Visualization import show_all_metrics
import time
import sys


def main():
    start = time.perf_counter()

    print("Starting Program")
    train_images, train_labels, test_images, test_labels, _, _, class_names = (
        load_and_preprocess_data(
            validation_split=0.2, one_hot=True, use_dataset="MNIST"
        )
    )
    print("Data loaded")
    train_images = train_images.reshape(-1, 1, 28, 28).astype(cp.float32)
    test_images = test_images.reshape(-1, 1, 28, 28).astype(cp.float32)

    sample_size = 10000
    train_images = train_images[:sample_size]
    train_labels = train_labels[:sample_size]

    model = Sequential(
        [
            Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Flatten(),  # -> (N, 256)
            Dense(256, 120),
            ReLU(),
            Dense(120, 84),
            ReLU(),
            Dense(84, 10),
            SoftmaxCrossEntropy(),
        ]
    )

    # model.load("cnn_final", )

    print("Beginning Training")
    epochs = 1
    learning_rate = 0.001
    history = []
    optimizer = use_optimizer(model.parameters(), type="Adam", lr=learning_rate)
    for epoch in range(epochs + 1):
        if epoch % 5 == 0 and epoch != 0:
            learning_rate *= 0.5
        if epoch != 0:
            model.train(optimizer, train_images, train_labels, batch_size=256)

        # Train Data
        epoch_train_predictions = cp.asnumpy(model.predict(train_images))
        epoch_train_labels_argmax = np.argmax(train_labels, axis=1)

        epoch_train_accuracy = np.mean(
            epoch_train_predictions == epoch_train_labels_argmax
        )
        epoch_train_loss = model.forward(train_images, train_labels)

        # Test data
        epoch_test_predictions = cp.asnumpy(model.predict(test_images))
        epoch_test_labels_argmax = np.argmax(test_labels, axis=1)

        epoch_test_accuracy = np.mean(
            epoch_test_predictions == epoch_test_labels_argmax
        )
        epoch_test_loss = model.forward(test_images, test_labels)

        history.append(
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": epoch_train_loss,
                "test_loss": epoch_test_loss,
                "train_acc": epoch_train_accuracy,
                "test_acc": epoch_test_accuracy,
            }
        )
        print(f"Test Accuracy Epoch {epoch}: {epoch_test_accuracy * 100:.2f}%")
        model.save("cnn_final")

    dataframe = pd.DataFrame(history)

    print("Evaluating on test data:")
    test_output = model.forward(test_images, test_labels)
    test_predictions = cp.asnumpy(model.predict(test_images))
    test_loss = model.cross_entropy(test_labels, test_output)
    print(f"Test Loss: {test_loss}")

    # Calculate accuracy
    test_labels_argmax = np.argmax(test_labels, axis=1)
    accuracy = np.mean(test_predictions == test_labels_argmax)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")

    show_all_metrics(test_labels_argmax, test_predictions, dataframe, class_names)


if __name__ == "__main__":
    main()
