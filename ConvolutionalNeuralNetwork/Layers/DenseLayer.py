import cupy as cp


class Dense:
    def __init__(self, inputSize, outputSize):
        # Initializes a matrix of dimensons inputSize x outputSize of weights
        self.weights = cp.random.randn(inputSize, outputSize) * cp.sqrt(2.0 / inputSize)
        # Inititalizes an array of biases for each 'neuron'
        self.bias = cp.zeros(outputSize)

        # Placeholders
        self.dW = cp.zeros_like(self.weights)
        self.db = cp.zeros_like(self.bias)

    def forward(self, inputs):
        self.inputs = inputs
        self.linear_output = cp.dot(inputs, self.weights) + self.bias

        return self.linear_output

    def backward(self, error):
        grad_weights = cp.dot(self.inputs.T, error)
        grad_biases = cp.sum(error, axis=0)

        self.dW[:] = grad_weights
        self.db[:] = grad_biases

        # Gives error to be backpropogated to previous layer
        return cp.dot(error, self.weights.T)
