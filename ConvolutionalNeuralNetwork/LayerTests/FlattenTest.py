import unittest
import cupy as cp
import numpy as np
from ConvolutionalNeuralNetwork.Layers.FlattenLayer import Flatten


class TestFlatten(unittest.TestCase):
    def setUp(self):
        # Create a random 4D tensor: (N, C, H, W)
        cp.random.seed(0)
        self.N, self.C, self.H, self.W = 3, 2, 4, 5
        self.input = cp.random.randn(self.N, self.C, self.H, self.W).astype(cp.float32)
        self.layer = Flatten()

    def test_forward_shape(self):
        out = self.layer.forward(self.input)
        # D should be C*H*W
        D = self.C * self.H * self.W
        self.assertEqual(out.shape, (self.N, D))

    def test_forward_content(self):
        out = self.layer.forward(self.input)

        # Check that flattening preserves row-major order:
        # Compare to a numpy flatten of the same data
        expected = self.input.get().reshape(self.N, -1)
        np.testing.assert_array_equal(out.get(), expected)

    def test_input_shape_saved(self):
        _ = self.layer.forward(self.input)
        self.assertEqual(self.layer.input_shape, (self.N, self.C, self.H, self.W))

    def test_backward_shape(self):
        # First forward to set input_shape
        _ = self.layer.forward(self.input)

        # Create a gradient of shape (N, D)
        D = self.C * self.H * self.W
        grad_out = cp.random.randn(self.N, D).astype(cp.float32)
        dx = self.layer.backward(grad_out)

        # Should restore to original input shape
        self.assertEqual(dx.shape, (self.N, self.C, self.H, self.W))

    def test_backward_content(self):
        # Forward to set shape
        _ = self.layer.forward(self.input)

        # Create a known gradient, say 0,1,2,... per-example flattened
        D = self.C * self.H * self.W
        seq = cp.arange(self.N * D, dtype=cp.float32).reshape(self.N, D)
        dx = self.layer.backward(seq)

        # Should match reshaping that same sequence
        expected = seq.get().reshape(self.N, self.C, self.H, self.W)
        np.testing.assert_array_equal(dx.get(), expected)


if __name__ == "__main__":
    unittest.main()
