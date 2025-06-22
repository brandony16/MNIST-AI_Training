import cupy as cp


class FastLayer:
    def __init__(self, inputSize, outputSize, activation="relu"):
        # Initializes a matrix of dimensons inputSize x outputSize of weights
        self.weights = cp.random.randn(inputSize, outputSize) * cp.sqrt(2.0 / inputSize)
        # Inititalizes an array of biases for each 'neuron'
        self.bias = cp.zeros(outputSize)

        if activation == "sigmoid":
            self.activation = self.sigmoid
            self.act_deriv = self.sigmoid_derivative
        else:  # Default to relu
            self.activation = self.reLU
            self.act_deriv = self.reLU_deriv

        # Adam Optimization Vars
        # First moment vector m. Moving average of past gradients
        # Second moment vecter v. Moving average of the squared gradients
        self.m_weights = cp.zeros_like(self.weights)
        self.v_weights = cp.zeros_like(self.weights)

        self.m_biases = cp.zeros_like(self.bias)
        self.v_biases = cp.zeros_like(self.bias)

        self.time_step = 0  # Used to calculate gradient gt at current time step t

    def forward(self, inputs):
        self.inputs = inputs
        self.linear_output = cp.dot(inputs, self.weights) + self.bias

        self.output = self.activation(self.linear_output)
        return self.output

    def backward(self, error, learningRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # The gradient of the loss with respect to the pre-activation input of the current layer.
        delta = error * self.act_deriv(self.output)

        grad_weights = cp.dot(self.inputs.T, delta)
        grad_biases = cp.sum(delta, axis=0)

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
        return cp.dot(delta, self.weights.T)

    def reLU(self, x):
        return cp.maximum(0, x)

    def reLU_deriv(self, x):
        return cp.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-x))

    def sigmoid_derivative(self, x):
        # Input should be softmax(linear_outputs)
        return x * (1 - x)
