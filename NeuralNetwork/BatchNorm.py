import cupy as cp


class BatchNorm:
    def __init__(self, dim, momentum=0.9, eps=1e-5):
        self.dim = dim
        self.momentum = momentum
        self.eps = eps

        # Learnable parameters
        self.gamma = cp.ones((1, dim))
        self.beta = cp.zeros((1, dim))

        self.dgamma = cp.zeros_like(self.gamma)
        self.dbeta = cp.zeros_like(self.beta)

        # Running averages for inference
        self.running_mean = cp.zeros((1, dim))
        self.running_var = cp.ones((1, dim))

        # Internal state for backprop
        self.training = True
        self.cache = None

    def forward(self, x, training=False):
        if training:
            mean = cp.mean(x, axis=0, keepdims=True)
            var = cp.var(x, axis=0, keepdims=True)
            std = cp.sqrt(var + self.eps)
            x_norm = (x - mean) / std

            out = self.gamma * x_norm + self.beta

            # Update running stats
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )

            self.cache = (x, x_norm, mean, var, std)
            return out
        else:
            # Inference: use running statistics
            x_norm = (x - self.running_mean) / cp.sqrt(self.running_var + self.eps)
            return self.gamma * x_norm + self.beta

    def backward(self, dout):
        x, x_norm, mean, var, std = self.cache
        N = x.shape[0]

        dbeta = cp.sum(dout, axis=0, keepdims=True)
        dgamma = cp.sum(dout * x_norm, axis=0, keepdims=True)

        dx_norm = dout * self.gamma

        dvar = cp.sum(dx_norm * (x - mean) * -0.5 * (std**-3), axis=0, keepdims=True)
        dmean = cp.sum(-dx_norm / std, axis=0, keepdims=True) + dvar * cp.mean(
            -2 * (x - mean), axis=0, keepdims=True
        )

        dx = dx_norm / std + dvar * 2 * (x - mean) / N + dmean / N

        # Store gradients for optimizer
        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx
