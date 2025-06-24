import cupy as cp


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
        num_inputs, _, height, width = inputs.shape

        self.width = width
        self.height = height
        self.out_width = int(((width + 2 * self.p - self.k_size) // self.s) + 1)
        self.out_height = int(((height + 2 * self.p - self.k_size) // self.s) + 1)

        inputs_padded = cp.pad(
            inputs,
            # Dont pad num inputs or channels
            pad_width=((0, 0), (0, 0), (self.p, self.p), (self.p, self.p)),
            mode="constant",
            constant_values=0.0,
        )

        output_buffer = cp.zeros(
            (num_inputs, self.c_out, self.out_height, self.out_width),
            dtype=cp.float32,
        )

        N = inputs_padded.shape[0]
        H_out, W_out = self.out_height, self.out_width
        K, S = self.k_size, self.s

        for n in range(N):
            im2col_buffer = cp.zeros(
                (
                    self.c_in * self.k_size * self.k_size,
                    self.out_height * self.out_width,
                ),
                dtype=cp.float32,
            )
            col_idx = 0
            for i in range(H_out):
                for j in range(W_out):
                    x, y = j * S, i * S

                    patch = inputs_padded[n, :, y : y + K, x : x + K]

                    im2col_buffer[:, col_idx] = patch.reshape(-1)
                    col_idx += 1

            out_flat = cp.dot(self.w_flat, im2col_buffer)
            output_buffer[n] = (
                out_flat.reshape(self.c_out, H_out, W_out) + self.biases[:, None, None]
            )

        return output_buffer
