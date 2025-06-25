import cupy as cp


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, y=None):
        """
        If the last layer is a loss layer (e.g. SoftmaxCrossEntropy),
        it can take both logits and targets y to return a scalar loss.
        """
        out = x
        for layer in self.layers:
            # if it's the final loss layer and y is provided:
            if (
                layer.__class__.__name__ == "SoftmaxCrossEntropy"
                and hasattr(layer, "forward")
                and y is not None
            ):
                out = layer.forward(out, y)
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
                grad = (
                    layer.backward(grad) if grad is not None else layer.backward(None)
                )
        return grad

    def parameters(self):
        """
        Gather all (param, grad) pairs so optimizers can update them.
        Looks for .weights/.dW and .biases/.db on each layer.
        """
        params = []
        for layer in self.layers:
            if hasattr(layer, "weights"):
                params.append((layer.weights, layer.dW))
            if hasattr(layer, "biases"):
                params.append((layer.biases, layer.db))
        return params

    def predict(self, x):
        """Run forward through all but the loss layer, then take argmax."""
        out = x
        for layer in self.layers:
            # skip the loss layer if present
            if layer.__class__.__name__ == "SoftmaxCrossEntropy":
                break
            out = layer.forward(out)
        # assume out is logits
        return cp.argmax(out, axis=1)

    def train(self, optimizer, train_data, train_labels, batch_size=64):
        N = train_data.shape[0]

        perm = cp.random.permutation(N)
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            data_batch = train_data[idx]
            label_batch = train_labels[idx]

            loss = self.forward(data_batch, label_batch)
            
            optimizer.zero_grad()

            self.backward()

            optimizer.step()
