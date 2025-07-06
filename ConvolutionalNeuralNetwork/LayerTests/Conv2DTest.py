import unittest
import numpy as np
import cupy as cp

from ConvolutionalNeuralNetwork.Layers.Conv2DLayer import Conv2D
import time


def naive_conv2d_forward(x, W, b, stride=1, pad=0):
    N, C_in, H, W_in = x.shape
    C_out, _, K, _ = W.shape
    H_out = (H + 2 * pad - K) // stride + 1
    W_out = (W_in + 2 * pad - K) // stride + 1

    # pad input
    x_p = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")
    out = np.zeros((N, C_out, H_out, W_out), dtype=x.dtype)

    for n in range(N):
        for f in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h0 = i * stride
                    w0 = j * stride
                    patch = x_p[n, :, h0 : h0 + K, w0 : w0 + K]
                    out[n, f, i, j] = np.sum(patch * W[f]) + b[f]
    return out


def naive_conv2d_backward_input(d_out, W, x_shape, stride=1, pad=0):
    N, C_in, H, W_in = x_shape
    C_out, _, K, _ = W.shape
    _, _, H_out, W_out = d_out.shape

    x_p_grad = np.zeros((N, C_in, H + 2 * pad, W_in + 2 * pad), dtype=d_out.dtype)
    # scatter gradients
    for n in range(N):
        for f in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h0 = i * stride
                    w0 = j * stride
                    x_p_grad[n, :, h0 : h0 + K, w0 : w0 + K] += W[f] * d_out[n, f, i, j]
    # unpad
    return x_p_grad[:, :, pad : pad + H, pad : pad + W_in]


class TestConv2D(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        # small test: batch=2, C_in=1, H=W=5
        self.x_cpu = np.random.randn(2, 1, 5, 5).astype(np.float32)
        self.x_gpu = cp.array(self.x_cpu)
        # layer params
        self.layer = Conv2D(
            in_channels=1, out_channels=2, padding=1, kernel_size=3, stride=1
        )
        # override random init for reproducibility
        W0 = np.random.randn(2, 1, 3, 3).astype(np.float32) * 0.1
        b0 = np.random.randn(2).astype(np.float32) * 0.1
        self.layer.weights = cp.array(W0)
        self.layer.bias = cp.array(b0)

    def test_forward(self):
        y_gpu = self.layer.forward(self.x_gpu)

        y_cpu = naive_conv2d_forward(
            self.x_cpu,
            cp.asnumpy(self.layer.weights),
            cp.asnumpy(self.layer.bias),
            stride=self.layer.s,
            pad=self.layer.p,
        )
        self.assertEqual(y_gpu.shape, y_cpu.shape)
        np.testing.assert_allclose(cp.asnumpy(y_gpu), y_cpu, rtol=1e-5, atol=1e-5)

    def test_backward_bias(self):
        # simple d_out = ones
        y = self.layer.forward(self.x_gpu)
        d_out = cp.ones_like(y)

        _ = self.layer.backward(d_out)

        # db should equal sum of d_out over N,H,W
        expected_db = cp.asnumpy(d_out.sum(axis=(0, 2, 3)))
        np.testing.assert_allclose(cp.asnumpy(self.layer.db), expected_db)

    def test_backward_weight_numerical(self):
        # finite-diff a single weight
        eps = 1e-3
        idx = (1, 0, 1, 1)  # filter 1, channel 0, position (1,1)
        W_cpu = cp.asnumpy(self.layer.weights)
        b_cpu = cp.asnumpy(self.layer.bias)

        # baseline forward sum
        def loss(W_mod):
            # set perturbed weights
            self.layer.weights = cp.array(W_mod)
            out = self.layer.forward(self.x_gpu)
            return float(cp.asnumpy(out.sum()))

        orig = W_cpu[idx]
        W_cpu[idx] = orig + eps
        l1 = loss(W_cpu)
        W_cpu[idx] = orig - eps
        l2 = loss(W_cpu)
        numeric_grad = (l1 - l2) / (2 * eps)

        # reset weights
        W_cpu[idx] = orig
        self.layer.weights = cp.array(W_cpu)

        # compute analytical gradient
        y = self.layer.forward(self.x_gpu)
        d_out = cp.ones_like(y)
        _ = self.layer.backward(d_out)
        analytic_grad = float(cp.asnumpy(self.layer.dW)[idx])

        self.assertAlmostEqual(numeric_grad, analytic_grad, places=3)

    def test_backward_input(self):
        y = self.layer.forward(self.x_gpu)
        d_out = cp.ones_like(y)
        dx_gpu = self.layer.backward(d_out)
        dx_cpu = naive_conv2d_backward_input(
            cp.asnumpy(d_out),
            cp.asnumpy(self.layer.weights),
            self.x_cpu.shape,
            stride=self.layer.s,
            pad=self.layer.p,
        )
        np.testing.assert_allclose(cp.asnumpy(dx_gpu), dx_cpu, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
