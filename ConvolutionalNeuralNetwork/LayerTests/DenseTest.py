import unittest
from ConvolutionalNeuralNetwork.Layers.DenseLayer import Dense
import cupy as cp


class TestDenseLayer(unittest.TestCase):
    def setUp(self):
        cp.random.seed(0)
        # dimensions
        self.batch_size = 4
        self.in_features = 5
        self.out_features = 3

        # random input
        self.inputs = cp.random.randn(self.batch_size, self.in_features)

        # make a Dense layer with fixed weights & biases
        self.layer = Dense(self.in_features, self.out_features)

        # override random init for reproducibility
        self.layer.weights = cp.random.randn(self.in_features, self.out_features)
        self.layer.bias = cp.random.randn(self.out_features)
        self.layer.forward(self.inputs)

    def test_forward_shape_and_value(self):
        """Forward should compute x·W + b and have correct shape."""
        out = self.layer.forward(self.inputs)

        # shape check
        self.assertEqual(out.shape, (self.batch_size, self.out_features))

        # manual computation
        expected = cp.dot(self.inputs, self.layer.weights) + self.layer.bias
        cp.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    def test_backward_shape_and_grads(self):
        """Backward should return dX of correct shape and compute dW, db properly."""
        # pretend upstream gradient is all ones
        dout = cp.ones((self.batch_size, self.out_features))
        dx = self.layer.backward(dout)

        # shape of dx
        self.assertEqual(dx.shape, self.inputs.shape)

        # expected gradients:
        # dW = x^T · dout
        expected_dW = cp.dot(self.inputs.T, dout)

        # db = sum over batch of dout
        expected_db = cp.sum(dout, axis=0)

        # dx = dout · W^T
        expected_dx = cp.dot(dout, self.layer.weights.T)

        cp.testing.assert_allclose(self.layer.dW, expected_dW, rtol=1e-6, atol=1e-6)
        cp.testing.assert_allclose(self.layer.db, expected_db, rtol=1e-6, atol=1e-6)
        cp.testing.assert_allclose(dx, expected_dx, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
