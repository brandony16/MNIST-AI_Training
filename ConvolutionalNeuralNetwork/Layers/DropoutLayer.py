import cupy as cp


class Dropout:
    def __init__(self, p: float):
        """
        Dropout layer with probability p of dropping each unit.

        Uses inverted dropout: at train time, units are zeroed with probability p,
        and the remaining activations are scaled by 1/(1-p) so that you don't need
        to rescale at inference.

        Args:
            p (float): dropout probability in [0, 1). Fraction of units to drop.
        """
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be in [0, 1).")
        self.p = p
        self.mask = None

    def forward(self, inputs: cp.ndarray, training: bool = True) -> cp.ndarray:
        """
        Forward pass for dropout.

        During training, randomly zeroes some of the elements of x with probability p,
        and scales the rest by 1/(1-p). During inference, returns x unchanged.

        Args:
            inputs (cp.ndarray): input array of any shape.
            training (bool): whether in training mode. If False, dropout is bypassed.

        Returns:
            cp.ndarray: output array, same shape as inputs.
        """
        if training and self.p > 0:
            # Probability of keeping a value
            keep_prob = 1.0 - self.p

            # Scale by 1/(1-p) to keep expected activation the same
            self.mask = (cp.random.random_sample(inputs.shape) < keep_prob) / keep_prob

            return inputs * self.mask
        else:
            # Inference mode or p = 0: Do nothing
            self.mask = None
            return inputs

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        """
        Backward pass for dropout.

        Multiplies the gradient by the same mask used in forward.
        If forward was called in inference mode (mask is None), returns grad_output unchanged.

        Args:
            grad_output (cp.ndarray): gradient, same shape as forward input.

        Returns:
            cp.ndarray: gradient to pass to previous layer.
        """
        if self.mask is not None:
            return grad_output * self.mask
        else:
            return grad_output
