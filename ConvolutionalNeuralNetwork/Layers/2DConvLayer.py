import cupy as cp
from cupy.lib.stride_tricks import sliding_window_view


class Conv2D:
    def __init__(self, in_channels, out_channels, padding=0, kernel_size=3, stride=1):

        self.c_in = in_channels
        self.c_out = out_channels
        self.p = padding
        self.k_size = kernel_size
        self.s = stride

        self.weights = cp.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        )
        self.w_flat = self.weights.reshape(self.c_out, -1)
        self.biases = cp.zeros((out_channels,), dtype=cp.float32)

    def forward(self, inputs):
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
            + self.biases[None, :, None, None]
        )

        return out

    def backward(self, d_out):
        N = d_out.shape[0]
        K, S, P = self.k_size, self.s, self.p

        # d_out is shape (N, C_out, H_out, W_out)
        # sum over batch and spatial dims
        d_bias = d_out.sum(axis=(0, 2, 3))

        # (c_out, c_in * k * k)
        d_out_flat = d_out.reshape(N, self.c_out, -1)  # (N, c_out, H_out*W_out)
        patches_T = self.im2col_batched.transpose(0, 2, 1)
        dW_batch = cp.matmul(d_out_flat, patches_T)
        dW_flat = dW_batch.sum(axis=0)

        self.dW = dW_flat.reshape(self.c_out, self.c_in, K, K)
        self.db = d_bias

        dx_padded = cp.zeros_like(self.inputs_padded)

        for n in range(N):
            d_out_flat = d_out[n].reshape(self.c_out, -1)  # (C_out, H_out*W_out)

            dx_cols = cp.dot(self.w_flat.T, d_out_flat)
            dx_cols = dx_cols.reshape(self.c_in, K, K, self.out_height, self.out_width)

            for i in range(self.out_height):
                for j in range(self.out_width):
                    y, x = i * S, j * S
                    # patch gradients for all channels at (i,j): shape (C_in, K, K)
                    patch_grad = dx_cols[:, :, :, i, j]
                    dx_padded[n, :, y : y + K, x : x + K] += patch_grad

        dx = dx_padded[:, :, P : P + self.height, P : P + self.width]
        return dx
