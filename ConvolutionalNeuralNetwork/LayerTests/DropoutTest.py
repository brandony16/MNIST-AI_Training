import unittest
import cupy as cp
from ConvolutionalNeuralNetwork.Layers.DropoutLayer import Dropout


class TestDropout(unittest.TestCase):
    def setUp(self):
        # fix seed for reproducibility
        cp.random.seed(42)
        self.input = cp.arange(1, 17, dtype=cp.float32).reshape((2, 2, 2, 2))

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            Dropout(-0.1)
        with self.assertRaises(ValueError):
            Dropout(1.0)
        # boundary valid
        Dropout(0.0)
        Dropout(0.9999)

    def test_forward_no_dropout(self):
        layer = Dropout(p=0.0)
        out = layer.forward(self.input, training=True)
        # no mask applied
        self.assertIsNone(layer.mask)
        cp.testing.assert_array_equal(out, self.input)

        # inference also unchanged
        out_inf = layer.forward(self.input, training=False)
        self.assertIsNone(layer.mask)
        cp.testing.assert_array_equal(out_inf, self.input)

    def test_forward_training_applies_mask(self):
        p = 0.25
        layer = Dropout(p=p)
        out = layer.forward(self.input, training=True)

        # mask should be stored and same shape
        self.assertIsNotNone(layer.mask)
        self.assertEqual(layer.mask.shape, self.input.shape)

        # mask values must be either 0 or 1/(1-p)
        keep_val = 1.0 / (1 - p)
        unique_vals = cp.unique(layer.mask)
        # convert to Python floats for comparison
        unique_vals = set(float(v) for v in unique_vals)
        self.assertTrue(unique_vals <= {0.0, keep_val})

        # check that zeros in mask zero out outputs, and nonzeros scale
        zero_positions = layer.mask == 0
        nonzero_positions = layer.mask == keep_val

        # zeros => output zero
        self.assertTrue(cp.all(out[zero_positions] == 0))
        # nonzeros => out = input * keep_val
        cp.testing.assert_allclose(
            out[nonzero_positions], self.input[nonzero_positions] * keep_val
        )

        # approximate drop rate
        drop_rate = int(cp.sum(zero_positions)) / self.input.size
        self.assertAlmostEqual(drop_rate, p, delta=0.1)

    def test_forward_inference_bypasses(self):
        p = 0.5
        layer = Dropout(p=p)
        out = layer.forward(self.input, training=False)
        # inference: no mask stored, input passes through
        self.assertIsNone(layer.mask)
        cp.testing.assert_array_equal(out, self.input)

    def test_backward_training_uses_same_mask(self):
        p = 0.3
        layer = Dropout(p=p)
        layer.forward(self.input, training=True)
        grad_out = cp.ones_like(self.input) * 2.0  # all twos

        dx = layer.backward(grad_out)

        # dx should equal grad_out * mask
        expected = grad_out * layer.mask
        cp.testing.assert_array_equal(dx, expected)

    def test_backward_inference_passes_gradient(self):
        layer = Dropout(p=0.4)
        # do an inference forward to clear mask
        _ = layer.forward(self.input, training=False)
        self.assertIsNone(layer.mask)

        grad_out = cp.arange(self.input.size, dtype=cp.float32).reshape(
            self.input.shape
        )
        dx = layer.backward(grad_out)
        # should be unchanged
        cp.testing.assert_array_equal(dx, grad_out)


if __name__ == "__main__":
    unittest.main()
