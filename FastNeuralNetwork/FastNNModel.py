from FastNeuralNetwork.FastLayer import FastLayer
import cupy as cp
from SoftmaxCELayer import SoftmaxCrossEntropyLayer
from DenseLayer import DenseLayer


class FastNeuralNetwork:
    def __init__(self, layer_sizes, activation="relu"):
        cp.random.seed(42)
        self.hidden_layers = []
        num_layers = len(layer_sizes)
        for i in range(num_layers - 2):
            self.hidden_layers.append(
                FastLayer(layer_sizes[i], layer_sizes[i + 1], activation)
            )

        # Final Dense (no activation) and CE layer for faster training
        self.final_dense = DenseLayer(layer_sizes[-2], layer_sizes[-1])
        self.ce_layer = SoftmaxCrossEntropyLayer()

    def forward(self, data, labels=None):
        data = cp.asarray(data)
        for layer in self.hidden_layers:
            data = layer.forward(data)
        logits = self.final_dense.forward(data)

        if labels is None:
            # inference: just return softmax probabilities
            z_max = cp.max(logits, axis=1, keepdims=True)
            exp_z = cp.exp(logits - z_max)
            return exp_z / exp_z.sum(axis=1, keepdims=True)

        return self.ce_layer.forward(logits, cp.asarray(labels))

    def backward(self, learningRate):
        gradient = self.ce_layer.backward()
        gradient = self.final_dense.backward(gradient, learningRate)

        for layer in reversed(self.hidden_layers):
            gradient = layer.backward(gradient, learningRate)

    def train(self, data, labels, epochs=10, learningRate=0.1, batch_size=32):
        data = cp.asarray(data)
        labels = cp.asarray(labels)

        num_samples = data.shape[0]
        for _ in range(epochs):
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
                self.forward(batch_data, batch_labels)
                self.backward(learningRate)

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
