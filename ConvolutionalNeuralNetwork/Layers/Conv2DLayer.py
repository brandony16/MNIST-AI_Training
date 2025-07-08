import cupy as cp
from cupy.lib.stride_tricks import sliding_window_view


class Conv2D:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        padding: int = 0,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        self.c_in = in_channels
        self.c_out = out_channels
        self.p = padding
        self.k_size = kernel_size
        self.s = stride

        # Initialize weights and biases using He initialization
        scale = cp.sqrt(2 / (in_channels * kernel_size * kernel_size))
        self.weights = (
            cp.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        )

        self.bias = cp.zeros((out_channels,), dtype=cp.float32)

        self.dW = cp.zeros_like(self.weights)
        self.db = cp.zeros_like(self.bias)

    @property
    def w_flat(self):
        return self.weights.reshape(self.c_out, -1)  # (Cout, Cin*K*K)

    def forward(self, inputs: cp.ndarray):
        N, _, height, width = inputs.shape
        K, S, P = self.k_size, self.s, self.p

        self.width = width
        self.height = height
        self.out_width = int(((width + 2 * P - K) // S) + 1)
        self.out_height = int(((height + 2 * P - K) // S) + 1)

        inputs_padded = cp.pad(
            inputs,
            # Dont pad num inputs or channels
            pad_width=((0, 0), (0, 0), (P, P), (P, P)),
            mode="constant",
            constant_values=0.0,
        )
        self.inputs_padded = inputs_padded

        # windows.shape == (N, C_in, H+2P−K+1, W+2P−K+1, K, K) (stride = 1)
        windows = sliding_window_view(inputs_padded, window_shape=(K, K), axis=(2, 3))
        windows = windows[:, :, ::S, ::S, :, :]  # Apply stride

        _, _, H_out, W_out, _, _ = windows.shape

        # Flatten each patch for im2col
        windows = windows.transpose(0, 1, 4, 5, 2, 3)
        im2col_batched = windows.reshape(N, self.c_in * K * K, H_out * W_out)
        self.im2col_batched = im2col_batched

        # bring the patch dim to the rightmost axis
        # shape == (N, H_out*W_out, C_in*K*K)
        im2col_t = im2col_batched.transpose(0, 2, 1)

        # shape == (N, H_out*W_out, C_out)
        out_flat = cp.dot(im2col_t, self.w_flat.T)

        out = (
            # (N, C_out, H_out, W_out)
            out_flat.reshape(N, H_out, W_out, self.c_out).transpose(0, 3, 1, 2)
            + self.bias[None, :, None, None]
        )

        return out

    def backward(self, d_out: cp.ndarray):
        N = d_out.shape[0]
        K, S, P = self.k_size, self.s, self.p
        H_out, W_out = self.out_height, self.out_width

        # d_out is shape (N, C_out, H_out, W_out)
        # sum over batch and spatial dims
        d_bias = d_out.sum(axis=(0, 2, 3))

        # (c_out, c_in * k * k)
        d_out_flat = d_out.reshape(N, self.c_out, -1)  # (N, c_out, H_out*W_out)
        patches_T = self.im2col_batched.transpose(0, 2, 1)
        dW_batch = cp.matmul(d_out_flat, patches_T)
        dW_flat = dW_batch.sum(axis=0)

        self.dW[:] = dW_flat.reshape(self.c_out, self.c_in, K, K)
        self.db[:] = d_bias

        # compute dx_cols: shape (N, C_in, K, K, H_out, W_out)
        d_out_flat = d_out.reshape(N, self.c_out, -1)  # (N, Cout, Hout*Wout)
        dx_cols_flat = cp.dot(
            d_out_flat.transpose(0, 2, 1), self.w_flat
        )  # (N, Hout*Wout, Cin*K*K)

        dx_cols = dx_cols_flat.reshape(N, H_out, W_out, self.c_in, K, K)

        # prepare padded gradient
        dx_padded = cp.zeros_like(self.inputs_padded)  # (N, Cin, H+2P, W+2P)

        # scatter each K×K patch back into dx_padded
        for i in range(H_out):
            for j in range(W_out):
                y0, x0 = i * S, j * S
                # slice of dx_padded: (N, Cin, K, K)
                patch_view = dx_padded[:, :, y0 : y0 + K, x0 : x0 + K]
                # the computed gradients: (N, Cin, K, K)
                grad_patch = dx_cols[:, i, j, :, :, :]
                patch_view += grad_patch

        # un‑pad
        return dx_padded[:, :, P : P + self.height, P : P + self.width]
