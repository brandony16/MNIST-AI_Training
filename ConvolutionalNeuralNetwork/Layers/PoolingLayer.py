import cupy as cp
from cupy.lib.stride_tricks import sliding_window_view


class MaxPool2D:
    def __init__(
        self,
        pool_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        pad: int | tuple[int, int] = 0,
    ):
        # normalize pool_size, stride, pad
        if isinstance(pool_size, int):
            self.pool_h = self.pool_w = pool_size
        else:
            self.pool_h, self.pool_w = pool_size
        if stride is None:
            self.stride_h, self.stride_w = self.pool_h, self.pool_w
        elif isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
        if isinstance(pad, int):
            self.pad_h = self.pad_w = pad
        else:
            self.pad_h, self.pad_w = pad

        # placeholders for caching
        self.x_padded = None
        self.windows = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward: extract sliding windows, take max over each.
        Caches both the padded input and sliding_window_view for backward.
        """
        # pad with -inf so that padded values never win
        self.x_padded = cp.pad(
            x,
            ((0, 0), (0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w)),
            mode="constant",
            constant_values=-cp.inf,
        )

        # windows shape: (N, C, H_out, W_out, pool_h, pool_w)
        windows = sliding_window_view(
            self.x_padded, (self.pool_h, self.pool_w), axis=(2, 3)
        )
        windows = windows[:, :, :: self.stride_h, :: self.stride_w, :, :]
        self.windows = windows

        # max over the last two dims
        return windows.max(axis=(4, 5))

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        """
        Backward: route grad_out back to the locations of maxima.
        We loop only over the pooled spatial dims (H_out, W_out).
        """
        N, C, H_out, W_out, ph, pw = self.windows.shape
        # initialize gradient w.r.t padded input
        dx_p = cp.zeros_like(self.x_padded)

        # expand grad_out for broadcasting
        # shape -> (N, C, H_out, W_out, 1, 1)
        go = grad_out.reshape(N, C, H_out, W_out, 1, 1)

        # for each spatial window, route gradient to max positions
        for i in range(H_out):
            for j in range(W_out):
                window = self.windows[:, :, i, j, :, :]  # (N,C,ph,pw)
                mask = window == window.max(axis=(2, 3), keepdims=True)
                dx_p[
                    :,
                    :,
                    i * self.stride_h : i * self.stride_h + ph,
                    j * self.stride_w : j * self.stride_w + pw,
                ] += (
                    mask * go[:, :, i, j, :, :]
                )

        # un-pad to original input shape
        _, _, H_pad, W_pad = self.x_padded.shape
        H = H_pad - 2 * self.pad_h
        W = W_pad - 2 * self.pad_w
        return dx_p[:, :, self.pad_h : self.pad_h + H, self.pad_w : self.pad_w + W]


class AvgPool2D:
    def __init__(
        self,
        pool_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        pad: int | tuple[int, int] = 0,
    ):
        # normalize pool_size, stride, pad
        if isinstance(pool_size, int):
            self.pool_h = self.pool_w = pool_size
        else:
            self.pool_h, self.pool_w = pool_size
        if stride is None:
            self.stride_h, self.stride_w = self.pool_h, self.pool_w
        elif isinstance(stride, int):
            self.stride_h = self.stride_w = stride
        else:
            self.stride_h, self.stride_w = stride
        if isinstance(pad, int):
            self.pad_h = self.pad_w = pad
        else:
            self.pad_h, self.pad_w = pad

        # placeholders for caching
        self.x_padded = None
        self.windows = None

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        """
        Forward: extract sliding windows, take max over each.
        Caches both the padded input and sliding_window_view for backward.
        """
        # pad with zeros (average of padded region should be zero)
        self.x_padded = cp.pad(
            x,
            ((0, 0), (0, 0), (self.pad_h, self.pad_h), (self.pad_w, self.pad_w)),
            mode="constant",
            constant_values=0.0,
        )

        # windows shape: (N, C, H_out, W_out, pool_h, pool_w)
        windows = sliding_window_view(
            self.x_padded, (self.pool_h, self.pool_w), axis=(2, 3)
        )
        windows = windows[:, :, :: self.stride_h, :: self.stride_w, :, :]
        self.windows = windows

        # max over the last two dims
        return windows.mean(axis=(4, 5))

    def backward(self, grad_out: cp.ndarray) -> cp.ndarray:
        """
        Backward: route grad_out back to the locations of maxima.
        We loop only over the pooled spatial dims (H_out, W_out).
        """
        N, C, H_out, W_out, ph, pw = self.windows.shape
        # initialize gradient w.r.t padded input
        dx_p = cp.zeros_like(self.x_padded)

        # expand grad_out for broadcasting
        # shape -> (N, C, H_out, W_out, 1, 1)
        go = grad_out.reshape(N, C, H_out, W_out, 1, 1)

        # each window has ph*pw inputs, so each gets go/(ph*pw)
        go = go / (ph * pw)

        for i in range(H_out):
            for j in range(W_out):
                dx_p[
                    :,
                    :,
                    i * self.stride_h : i * self.stride_h + ph,
                    j * self.stride_w : j * self.stride_w + pw,
                ] += go[:, :, i, j, :, :]

        # un-pad to original input shape
        _, _, H_pad, W_pad = self.x_padded.shape
        H = H_pad - 2 * self.pad_h
        W = W_pad - 2 * self.pad_w
        return dx_p[:, :, self.pad_h : self.pad_h + H, self.pad_w : self.pad_w + W]
