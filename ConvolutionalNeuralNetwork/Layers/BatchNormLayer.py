import cupy as cp


class BatchNorm2D:
    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        """
        2D Batch Normalization layer

        Args:
            num_features (int): Number of channels C.
            momentum (float): Momentum for running mean/var update (in [0,1)).
            eps (float): Small constant to avoid divide-by-zero.
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps

        # Learnable scale and shift parameters, one per channel
        self.gamma = cp.ones((num_features,), dtype=cp.float32)
        self.beta = cp.zeros((num_features,), dtype=cp.float32)

        # Temp values for optimizer param list
        self.dgamma = cp.zeros_like(self.gamma)
        self.dbeta = cp.zeros_like(self.beta)

        # Running estimates used for evaluation
        self.running_mean = cp.zeros((num_features,), dtype=cp.float32)
        self.running_var = cp.ones((num_features,), dtype=cp.float32)

        # Buffers to store intermediate results for backward
        self.cache = {}

    def forward(self, x: cp.ndarray, training: bool = True) -> cp.ndarray:
        """
        Forward pass.

        During training:
        - Compute batch mean/var over N*H*W for each channel.
        - Normalize and scale/shift.
        - Update running estimates.

        During eval:
        - Use stored running_mean/running_var instead of batch stats. No updates

        Args:
            daxta (cp.ndarray): Input of shape (N, C, H, W).
            training (bool): If False, skips batch-stat computation.

        Returns:
            cp.ndarray: Same shape as x.
        """
        # Batch size, channels, height, width
        N, C, H, W = x.shape
        assert C == self.num_features

        if training:
            # compute per-channel mean & var
            mean = cp.mean(x, axis=(0, 2, 3))  # shape (C,)
            var = cp.var(x, axis=(0, 2, 3))  # shape (C,)

            # update running stats
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            mean = self.running_mean
            var = self.running_var

        # reshape for broadcasting: (1, C, 1, 1)
        mean_b = mean.reshape(1, C, 1, 1)
        var_b = var.reshape(1, C, 1, 1)
        gamma_b = self.gamma.reshape(1, C, 1, 1)
        beta_b = self.beta.reshape(1, C, 1, 1)

        # normalize
        inv_std = 1.0 / cp.sqrt(var_b + self.eps)
        x_centered = x - mean_b
        x_hat = x_centered * inv_std
        out = gamma_b * x_hat + beta_b

        # cache for backward
        if training:
            self.cache = {
                "x_hat": x_hat,
                "inv_std": inv_std,
                "gamma": gamma_b,
                "mean_b": mean_b,
                "var_b": var_b,
                "N_HW": N * H * W,
            }

        return out

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        """
        Backward pass.

        Computes gradients w.r.t.:
         - gamma and beta (learnable parameters)
         - input x

        Returns:
            cp.ndarray: Gradient w.r.t. input x, same shape as grad_out.
        """
        x_hat = self.cache["x_hat"]
        inv_std = self.cache["inv_std"]
        gamma_b = self.cache["gamma"]
        N_HW = self.cache["N_HW"]

        # Gradients for scale and shift
        # sum over (N, H, W) for each channel
        dgamma = cp.sum(grad_out * x_hat, axis=(0, 2, 3))
        dbeta = cp.sum(grad_out, axis=(0, 2, 3))

        # store parameter gradients
        self.dgamma = dgamma
        self.dbeta = dbeta

        # Gradient wrt normalized x
        dx_hat = grad_out * gamma_b

        # Vectorized formula for dX:
        # See: https://arxiv.org/pdf/1502.03167.pdf, Section on BatchNorm backward
        term1 = dx_hat
        term2 = cp.mean(dx_hat, axis=(0, 2, 3), keepdims=True)
        term3 = x_hat * cp.mean(dx_hat * x_hat, axis=(0, 2, 3), keepdims=True)
        dx = (inv_std / N_HW) * (N_HW * term1 - term2 - term3)

        return dx
