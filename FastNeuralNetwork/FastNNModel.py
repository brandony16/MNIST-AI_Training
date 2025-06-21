import numpy as np
from FastNeuralNetwork.FastLayer import FastLayer
import cupy as cp

class FastNeuralNetwork:
    def __init__(self, layer_sizes, activation="relu"):
        cp.random.seed(42)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            activation = activation if i < len(layer_sizes) - 2 else "softmax"
            self.layers.append(FastLayer(layer_sizes[i], layer_sizes[i + 1], activation))

    def forward(self, data):
        data = cp.asarray(data)
        for layer in self.layers:
            data = layer.forward(data)
        return data

    def backward(self, y, output, learningRate):
        error = self.cross_entropy_derivative(y, output)
        for layer in reversed(self.layers):
            error = layer.backward(error, learningRate)

    def train(self, data, labels, epochs=10, learningRate=0.1, batch_size=32):
        data = cp.asarray(data)
        labels = cp.asarray(labels)

        num_samples = data.shape[0]
        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            indices = cp.arange(num_samples)
            cp.random.shuffle(indices)
            data = data[indices]
            labels = labels[indices]
            print("Epoch Started")

            # Process the data in batches
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_data = data[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]

                # Forward and backward pass for the batch
                output = self.forward(batch_data)
                self.backward(batch_labels, output, learningRate)

            if epoch % epochs // 10 == 0:
                print(f"Epoch {epoch} of {epochs} completed")
                print(
                    f"Training Loss: {self.cross_entropy(labels, self.forward(data))}"
                )

    # Loss calculation
    def cross_entropy(self, y, output):
        # Clip values to prevent log(0)
        output = cp.asarray(output)
        y = cp.asarray(y)
        output = cp.clip(output, 1e-12, 1.0 - 1e-12)
        return -cp.mean(cp.sum(y * cp.log(output + 1e-8), axis=1))

    def cross_entropy_derivative(self, y, output):
        return output - y

    def predict(self, data):
        data = cp.asarray(data)
        output = self.forward(data)
        return cp.argmax(output, axis=1)
