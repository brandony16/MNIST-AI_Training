class Flatten:
    def forward(self, inputs):
        """
        inputs: cupy array of shape (N, d1, d2, ..., dk)
        returns: array of shape (N, D) where D = d1*d2*...*dk
        """
        self.input_shape = inputs.shape

        N = inputs.shape[0]
        D = 1
        for dim in inputs.shape[1:]:
            D *= dim

        return inputs.reshape(N, D)

    def backward(self, gradient_output):
        """
        gradient_output: cupy array of shape (N, D)
        returns: gradient reshaped back to original input shape
        """

        return gradient_output.reshape(self.input_shape)
