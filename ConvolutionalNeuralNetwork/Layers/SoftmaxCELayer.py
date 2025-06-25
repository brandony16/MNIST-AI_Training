import cupy as cp


class SoftmaxCrossEntropy:
    def forward(self, logits, labels):
        # Softmax
        max_output = cp.max(logits, axis=1, keepdims=True)
        exp_out = cp.exp(logits - max_output)
        self.probabilities = exp_out / cp.sum(exp_out, axis=1, keepdims=True)

        self.labels = labels
        return self.cross_entropy(labels, self.probabilities)

    def backward(self):
        batch_size = self.labels.shape[0]
        delta = (self.probabilities - self.labels) / batch_size
        return delta

    def cross_entropy(self, labels, output):
        # Clip values to prevent log(0)
        output = cp.clip(output, 1e-12, 1.0 - 1e-12)
        self.loss = -cp.mean(cp.sum(labels * cp.log(output + 1e-8), axis=1))
        return self.loss
