import cupy as cp
from SoftmaxCELayer import SoftmaxCrossEntropy
from DenseLayer import Dense
from ActivationLayers import get_activation
from FlattenLayer import Flatten
from DropoutLayer import Dropout


class NeuralNetwork:
    def __init__(
        self,
        layer_sizes,
        activation="relu",
    ):
        cp.random.seed(42)

        # Build layers
        self.layers = []
        num_layers = len(layer_sizes)
        self.layers.append(Flatten())
        for i in range(num_layers - 1):
            self.layers.append(Dense(layer_sizes[i], layer_sizes[i + 1]))

            # Dont add activation after final Dense layer
            if i != num_layers - 2:
                self.layers.append(get_activation(activation)())
                self.layers.append(Dropout(0.2))
        self.layers.append(SoftmaxCrossEntropy())

        self._training = False

    def train_mode(self):
        self._training = True

    def eval_mode(self):
        self._training = False

    def forward(self, data, labels=None):
        out = cp.asarray(data)
        for layer in self.layers:
            name = layer.__class__.__name__
            # if it's the final loss layer and y is provided:
            if name == "SoftmaxCrossEntropy":
                if labels is not None:
                    out = layer.forward(out, cp.asarray(labels))
            elif name == "Dropout":
                out = layer.forward(out, training=self._training)
            else:
                out = layer.forward(out)
        return out

    def backward(self):
        grad = None
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                grad = layer.backward(grad) if grad is not None else layer.backward()

        return grad

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "weights"):
                params.append((layer.weights, layer.dW))
            if hasattr(layer, "bias"):
                params.append((layer.bias, layer.db))
        return params

    def train(self, optimizer, data, labels, batch_size=64, augment_fn=None):
        """Trains the model for one epoch on the given training data"""
        self.train_mode()
        N = data.shape[0]

        train_data = cp.asarray(data)
        train_labels = cp.asarray(labels)

        perm = cp.random.permutation(N)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            data_batch = train_data[idx]
            label_batch = train_labels[idx]

            if augment_fn is not None:
                data_batch = augment_fn(data_batch, max_shift=2)

            optimizer.zero_grad()

            loss = self.forward(data_batch, label_batch)

            self.backward()

            optimizer.step()

    def predict(self, data, batch_size=1024):
        out = cp.asarray(data)
        N = out.shape[0]
        all_preds = []
        for i in range(0, N, batch_size):
            batch_pred = out[i : i + batch_size]
            for layer in self.layers:
                name = layer.__class__.__name__
                # skip the loss layer
                if name == "SoftmaxCrossEntropy":
                    break
                if name == "Dropout":
                    batch_pred = layer.forward(batch_pred, training=False)
                else:
                    batch_pred = layer.forward(batch_pred)

            preds = cp.argmax(batch_pred, axis=1)
            all_preds.append(preds)

        return cp.concatenate(all_preds, axis=0)
