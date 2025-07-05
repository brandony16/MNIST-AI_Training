import numpy as np
import cupy as cp


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, y=None):
        """
        If the last layer is a loss layer (e.g. SoftmaxCrossEntropy),
        it can take both logits and targets y to return a scalar loss.
        """
        out = cp.asarray(x)
        self.forward_times = []
        for layer in self.layers:
            # if it's the final loss layer and y is provided:
            if (
                layer.__class__.__name__ == "SoftmaxCrossEntropy"
                and hasattr(layer, "forward")
                and y is not None
            ):
                out = layer.forward(out, cp.asarray(y))
            else:
                out = layer.forward(out)
        return out

    def backward(self):
        """
        Call backward on the final layer (which must have stored its dy/dx),
        then propagate through all preceding layers in reverse.
        """
        grad = None
        for layer in reversed(self.layers):
            if hasattr(layer, "backward"):
                grad = layer.backward(grad) if grad is not None else layer.backward()
        return grad

    def parameters(self):
        """
        Gather all (param, grad) pairs so optimizers can update them.
        Looks for .weights/.dW and .bias/.db on each layer.
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, "weights"):
                params.append((layer.weights, layer.dW))
            if hasattr(layer, "bias"):
                params.append((layer.bias, layer.db))
        return params

    def predict(self, x, batch_size=256):
        """Run forward through all but the loss layer, then take argmax."""
        out = cp.asarray(x)
        N = out.shape[0]
        all_preds = []
        for i in range(0, N, batch_size):
            batch_pred = out[i : i + batch_size]
            for layer in self.layers:
                # skip the loss layer if present
                if layer.__class__.__name__ == "SoftmaxCrossEntropy":
                    break
                batch_pred = layer.forward(batch_pred)

            preds = cp.argmax(batch_pred, axis=1)
            all_preds.append(preds)

        return cp.concatenate(all_preds, axis=0)

    def train(self, optimizer, train_data, train_labels, batch_size=64):
        N = train_data.shape[0]

        train_data = cp.asarray(train_data)
        train_labels = cp.asarray(train_labels)

        perm = cp.random.permutation(N)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            data_batch = train_data[idx]
            label_batch = train_labels[idx]

            loss = self.forward(data_batch, label_batch)

            if i != 0:
                optimizer.zero_grad()

            self.backward()

            optimizer.step()

    def save(self, path):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                params[f"layer{i}_W"] = layer.weights.get()
            if hasattr(layer, "bias"):
                params[f"layer{i}_B"] = layer.bias.get()

        np.savez_compressed(path, **params)

    @classmethod
    def load(cls, path, *layer_constructors):
        """
        Reconstruct a Sequential model from the given layer constructors
        and load weights from `path + ".npz"`.

        `layer_constructors` should be the exact same list you passed to __init__,
        _not_ instances but callables, e.g.:
            Conv2D, ReLU, MaxPool2D, ..., Dense, SoftmaxCrossEntropy
        """
        # 1) Build fresh layers
        layers = [ctor() for ctor in layer_constructors]
        model = cls(layers)

        # 2) Load saved params
        data = np.load(path + ".npz")
        for idx, layer in enumerate(model.layers):
            w_key = f"layer{idx}_W"
            b_key = f"layer{idx}_B"
            if w_key in data and b_key in data:
                # assign back to GPU arrays
                layer.weights[:] = cp.asarray(data[w_key])
                layer.bias[:] = cp.asarray(data[b_key])
        print(f"[Loaded] weights from {path}.npz")
        return model
