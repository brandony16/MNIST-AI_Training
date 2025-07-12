import numpy as np
import cupy as cp


class Sequential:
    def __init__(self, layers):
        self.layers = layers
        self._training = False

    def train_mode(self):
        """Put model in training mode (BatchNorm updates, Dropout active)."""
        self._training = True

    def eval(self):
        """Put model in eval mode (BatchNorm fixed, Dropout bypassed)."""
        self._training = False

    def forward(self, x, y=None):
        """
        If the last layer is a loss layer (e.g. SoftmaxCrossEntropy),
        it can take both logits and targets y to return a scalar loss.
        """
        out = cp.asarray(x)
        self.forward_times = []
        for layer in self.layers:
            name = layer.__class__.__name__
            # if it's the final loss layer and y is provided:
            if name == "SoftmaxCrossEntropy":
                if y is not None:
                    out = layer.forward(out, cp.asarray(y))
            elif name == "BatchNorm2D" or name == "Dropout":
                out = layer.forward(out, training=self._training)
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
            if hasattr(layer, "gamma"):
                params.append((layer.gamma, layer.dgamma))
                params.append((layer.beta, layer.dbeta))
        return params

    def predict(self, x, y, batch_size=1024):
        """Run forward through all but the loss layer, then take argmax."""
        N = x.shape[0]
        all_preds = []
        total_loss = 0.0
        total_count = 0

        for i in range(0, N, batch_size):
            xb = cp.asarray(x[i : i + batch_size])
            yb = cp.asarray(y[i : i + batch_size])

            out = xb
            # forward through all but loss layer
            for layer in self.layers:
                name = layer.__class__.__name__
                if name == "SoftmaxCrossEntropy":
                    break
                if name in ("BatchNorm2D", "Dropout"):
                    out = layer.forward(out, training=False)
                else:
                    out = layer.forward(out)

            # out now contains logits for this batch
            # 1) compute batch predictions
            batch_preds = cp.argmax(out, axis=1)
            all_preds.append(batch_preds)

            # 2) compute batch loss via the final layer
            #    we assume the last layer is SoftmaxCrossEntropy
            loss_layer = next(
                l for l in self.layers if l.__class__.__name__ == "SoftmaxCrossEntropy"
            )
            batch_loss = loss_layer.forward(out, yb)  # returns scalar loss
            total_loss += float(batch_loss) * xb.shape[0]
            total_count += xb.shape[0]

        preds = cp.concatenate(all_preds, axis=0)
        avg_loss = total_loss / total_count
        return preds, avg_loss

    def train(
        self,
        optimizer,
        train_data,
        train_labels,
        batch_size=64,
        augment_fn=None,
        scheduler=None,
    ):
        """Trains the model for one epoch on the given training data"""
        self.train_mode()
        N = train_data.shape[0]

        train_data = cp.asarray(train_data)
        train_labels = cp.asarray(train_labels)

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

            self.clip_gradients(max_norm=2.0)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

    def clip_gradients(self, max_norm):
        all_grads = []

        for layer in self.layers:
            # Standard weights and biases
            if hasattr(layer, "dW"):
                all_grads.append(layer.dW.ravel())
            if hasattr(layer, "db"):
                all_grads.append(layer.db.ravel())

            # BatchNorm parameters
            if hasattr(layer, "dgamma"):
                all_grads.append(layer.dgamma.ravel())
            if hasattr(layer, "dbeta"):
                all_grads.append(layer.dbeta.ravel())

        if not all_grads:
            return

        flat_grads = np.concatenate(all_grads)
        total_norm = np.linalg.norm(flat_grads)

        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for layer in self.layers:
                if hasattr(layer, "dW"):
                    layer.dW *= scale
                if hasattr(layer, "db"):
                    layer.db *= scale
                if hasattr(layer, "dgamma"):
                    layer.dgamma *= scale
                if hasattr(layer, "dbeta"):
                    layer.dbeta *= scale
