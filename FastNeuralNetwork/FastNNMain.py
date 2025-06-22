import numpy as np
import cupy as cp
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from loadMNIST import load_and_preprocess_mnist
from FastNeuralNetwork.FastNNModel import FastNeuralNetwork
from Visualization import plot_curves, show_confusion_matrix, show_metrics
import time


def main():
    print("Starting Program")
    train_images, train_labels, test_images, test_labels = load_and_preprocess_mnist()
    # Define the neural network architecture
    layer_sizes = [784, 512, 248, 10]

    # relu and sigmoid activation functions available
    nn = FastNeuralNetwork(layer_sizes, "relu")
    print("Beginning Training")
    epochs = 10
    learning_rate = 0.001
    history = []
    for epoch in range(epochs + 1):
        if epoch % 3 == 0 and epoch != 0:
            learning_rate *= 0.5
        if epoch != 0:
            nn.train(train_images, train_labels, 1, learning_rate, 128)

        # Train Data
        epoch_train_output = nn.forward(train_images)
        epoch_train_predictions = cp.asnumpy(nn.predict(train_images))
        epoch_train_labels_argmax = np.argmax(train_labels, axis=1)

        epoch_train_accuracy = np.mean(
            epoch_train_predictions == epoch_train_labels_argmax
        )
        epoch_train_loss = nn.cross_entropy(train_labels, epoch_train_output)

        # Test data
        epoch_test_output = nn.forward(test_images)
        epoch_test_predictions = cp.asnumpy(nn.predict(test_images))
        epoch_test_labels_argmax = np.argmax(test_labels, axis=1)

        epoch_test_accuracy = np.mean(
            epoch_test_predictions == epoch_test_labels_argmax
        )
        epoch_test_loss = nn.cross_entropy(test_labels, epoch_test_output)

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

    dataframe = pd.DataFrame(history)
    plot_curves(dataframe)

    print("Evaluating on test data:")
    test_output = nn.forward(test_images)
    test_predictions = cp.asnumpy(nn.predict(test_images))
    test_loss = nn.cross_entropy(test_labels, test_output)
    print(f"Test Loss: {test_loss}")

    # Calculate accuracy
    test_labels_argmax = np.argmax(test_labels, axis=1)
    accuracy = np.mean(test_predictions == test_labels_argmax)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    show_confusion_matrix(test_labels_argmax, test_predictions)
    show_metrics(test_labels_argmax, test_predictions)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} seconds")
