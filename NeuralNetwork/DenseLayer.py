import cupy as cp


class DenseLayer:
    def __init__(self, inputSize, outputSize):
        # Initializes a matrix of dimensons inputSize x outputSize of weights
        self.weights = cp.random.randn(inputSize, outputSize) * cp.sqrt(2.0 / inputSize)
        # Inititalizes an array of biases for each 'neuron'
        self.bias = cp.zeros(outputSize)

        # Adam Optimization Vars
        self.m_weights = cp.zeros_like(
            self.weights
        )  # First moment vector m. Moving average of past gradients
        self.v_weights = cp.zeros_like(
            self.weights
        )  # Second moment vecter v. Moving average of the squared gradients
        self.m_biases = cp.zeros_like(self.bias)
        self.v_biases = cp.zeros_like(self.bias)
        self.time_step = 0  # Used to calculate gradient gt at current time step t

    def forward(self, inputs):
        self.inputs = inputs
        self.linear_output = cp.dot(inputs, self.weights) + self.bias

        return self.linear_output

    def backward(self, error, learningRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        grad_weights = cp.dot(self.inputs.T, error)
        grad_biases = cp.sum(error, axis=0)

        self.time_step += 1

        # First moment vector. Beta is decay rate. Higher beta gives more weight toward previous gradients
        self.m_weights *= beta1
        self.m_weights += (1 - beta1) * grad_weights
        self.m_biases *= beta1
        self.m_biases += (1 - beta1) * grad_biases

        # Second moment vector.
        self.v_weights *= beta2
        self.v_weights += (1 - beta2) * (grad_weights**2)
        self.v_biases *= beta2
        self.v_biases += (1 - beta2) * (grad_biases**2)

        # To correct the biases introduced by initializing the first and second moment vectors at 0, Adam computes bias-corrected estimates
        m_weights_hat = self.m_weights / (1 - beta1**self.time_step)
        m_biases_hat = self.m_biases / (1 - beta1**self.time_step)

        v_weights_hat = self.v_weights / (1 - beta2**self.time_step)
        v_biases_hat = self.v_biases / (1 - beta2**self.time_step)

        # Update weights and biases using formula. Epsilon prevents divison by 0
        self.weights -= (
            learningRate * m_weights_hat / (cp.sqrt(v_weights_hat) + epsilon)
        )

        self.bias -= learningRate * m_biases_hat / (cp.sqrt(v_biases_hat) + epsilon)

        # Gives error to be backpropogated to previous layer
        return cp.dot(error, self.weights.T)
