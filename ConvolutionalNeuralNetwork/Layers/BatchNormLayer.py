import cupy as cp


class BatchNorm2D:
    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        """
        2D Batch Normalization layer for CNN feature maps.

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

        # Running estimates used for evaluation
        self.running_mean = cp.zeros((num_features,), dtype=cp.float32)
        self.running_var = cp.ones((num_features,), dtype=cp.float32)

        # Buffers to store intermediate results for backward
        self.cache = {}
