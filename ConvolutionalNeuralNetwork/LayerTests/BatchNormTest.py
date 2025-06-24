import unittest
import cupy as cp
from ConvolutionalNeuralNetwork.Layers.BatchNormLayer import BatchNorm2D


class TestBatchNorm2D(unittest.TestCase):
    def setUp(self):
        self.N, self.C, self.H, self.W = 4, 3, 5, 5
        self.layer = BatchNorm2D(num_features=self.C)
        self.x = cp.random.randn(self.N, self.C, self.H, self.W).astype(cp.float32)

    def test_output_shape(self):
        out = self.layer.forward(self.x, training=True)
        self.assertEqual(out.shape, self.x.shape)

    def test_training_mean_variance(self):
        out = self.layer.forward(self.x, training=True)
        mean = cp.mean(out, axis=(0, 2, 3))
        std = cp.std(out, axis=(0, 2, 3))
        
        cp.testing.assert_allclose(mean, cp.zeros_like(mean), atol=1e-4, rtol=1e-3)
        cp.testing.assert_allclose(std, cp.ones_like(std), atol=1e-4, rtol=1e-3)

    def test_inference_uses_running_stats(self):
        # Do a training forward to update running stats
        self.layer.forward(self.x, training=True)
        running_mean = self.layer.running_mean.copy()
        running_var = self.layer.running_var.copy()

        # New random input & forward pass
        x2 = cp.random.randn(self.N, self.C, self.H, self.W).astype(cp.float32)
        out_eval = self.layer.forward(x2, training=False)

        # Manually compute what BatchNorm should do at inference
        # y = gamma * (x2 - running_mean) / sqrt(running_var + eps) + beta
        rm = running_mean.reshape(1, self.C, 1, 1)
        rv = running_var.reshape(1, self.C, 1, 1)
        gm = self.layer.gamma.reshape(1, self.C, 1, 1)
        bm = self.layer.beta.reshape(1, self.C, 1, 1)
        inv_std = 1.0 / cp.sqrt(rv + self.layer.eps)
        expected = gm * (x2 - rm) * inv_std + bm

        cp.testing.assert_allclose(out_eval, expected, atol=1e-6, rtol=1e-5)

    def test_backward_pass(self):
        out = self.layer.forward(self.x, training=True)
        grad_out = cp.random.randn(*out.shape).astype(cp.float32)
        dx = self.layer.backward(grad_out)

        # Shape checks
        self.assertEqual(dx.shape, self.x.shape)
        self.assertEqual(self.layer.dgamma.shape, (self.C,))
        self.assertEqual(self.layer.dbeta.shape, (self.C,))

        # Make sure gradients aren't NaNs
        self.assertFalse(cp.isnan(cp.sum(dx)))
        self.assertFalse(cp.isnan(cp.sum(self.layer.dgamma)))
        self.assertFalse(cp.isnan(cp.sum(self.layer.dbeta)))


if __name__ == "__main__":
    unittest.main()
