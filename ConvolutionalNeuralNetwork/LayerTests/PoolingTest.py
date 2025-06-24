import unittest
import numpy as np
import cupy as cp

# adjust this import to match your code’s location
from ConvolutionalNeuralNetwork.Layers.PoolingLayer import MaxPool2D


def naive_maxpool2d_forward(x, pool_h, pool_w, stride_h, stride_w, pad_h, pad_w):
    N, C, H, W = x.shape
    # add padding
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=-np.inf,
    )
    out_h = (H + 2 * pad_h - pool_h) // stride_h + 1
    out_w = (W + 2 * pad_w - pool_w) // stride_w + 1
    out = np.empty((N, C, out_h, out_w), dtype=x.dtype)
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride_h
                    w_start = j * stride_w
                    window = x_padded[
                        n, c, h_start : h_start + pool_h, w_start : w_start + pool_w
                    ]
                    out[n, c, i, j] = np.max(window)
    return out


def naive_maxpool2d_backward(
    x, pool_h, pool_w, stride_h, stride_w, pad_h, pad_w, grad_out
):
    N, C, H, W = x.shape
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=-np.inf,
    )
    dx_padded = np.zeros_like(x_padded)
    out_h, out_w = grad_out.shape[2], grad_out.shape[3]
    for n in range(N):
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * stride_h
                    w_start = j * stride_w
                    window = x_padded[
                        n, c, h_start : h_start + pool_h, w_start : w_start + pool_w
                    ]
                    # find mask of max positions (could be multiple)
                    mask = window == np.max(window)
                    dx_padded[
                        n, c, h_start : h_start + pool_h, w_start : w_start + pool_w
                    ] += (mask * grad_out[n, c, i, j])
    # remove padding
    if pad_h or pad_w:
        return dx_padded[:, :, pad_h : pad_h + H, pad_w : pad_w + W]
    else:
        return dx_padded


class TestMaxPool2D(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        # random small input
        self.x_cpu = np.random.randn(2, 3, 5, 6).astype(np.float32)
        self.x_gpu = cp.array(self.x_cpu)
        # pooling parameters
        self.pool_size = (2, 2)
        self.stride = (2, 2)
        self.pad = (0, 0)
        # initialize your layer
        self.layer = MaxPool2D(
            pool_size=self.pool_size, stride=self.stride, pad=self.pad
        )

    def test_forward(self):
        # forward on GPU
        y_gpu = self.layer.forward(self.x_gpu)
        y_cpu = naive_maxpool2d_forward(
            self.x_cpu,
            pool_h=self.pool_size[0],
            pool_w=self.pool_size[1],
            stride_h=self.stride[0],
            stride_w=self.stride[1],
            pad_h=self.pad[0],
            pad_w=self.pad[1],
        )
        # compare shapes and values
        self.assertEqual(y_gpu.shape, y_cpu.shape)
        np.testing.assert_allclose(cp.asnumpy(y_gpu), y_cpu, rtol=1e-6, atol=1e-6)

    def test_backward(self):
        # do forward to cache descriptors
        y_gpu = self.layer.forward(self.x_gpu)
        # create upstream gradient of ones
        grad_out = cp.ones_like(y_gpu)
        dx_gpu = self.layer.backward(grad_out)
        # compute naive backward on CPU
        dx_cpu = naive_maxpool2d_backward(
            self.x_cpu,
            pool_h=self.pool_size[0],
            pool_w=self.pool_size[1],
            stride_h=self.stride[0],
            stride_w=self.stride[1],
            pad_h=self.pad[0],
            pad_w=self.pad[1],
            grad_out=cp.asnumpy(grad_out),
        )
        # compare shapes and values
        self.assertEqual(dx_gpu.shape, dx_cpu.shape)
        np.testing.assert_allclose(cp.asnumpy(dx_gpu), dx_cpu, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
