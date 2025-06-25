import cupy as cp


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters  # List of (param, grad) tuples
        self.lr = lr

    def step(self):
        for param, grad in self.parameters:
            param -= self.lr * grad

    def zero_grad(self):
        for _, grad in self.parameters:
            grad.fill(0.0)


class Adam:
    def __init__(self, learningRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
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

    def step(self):
        # # First moment vector. Beta is decay rate. Higher beta gives more weight toward previous gradients
        # self.m_weights *= beta1
        # self.m_weights += (1 - beta1) * grad_weights
        # self.m_biases *= beta1
        # self.m_biases += (1 - beta1) * grad_biases

        # # Second moment vector.
        # self.v_weights *= beta2
        # self.v_weights += (1 - beta2) * (grad_weights**2)
        # self.v_biases *= beta2
        # self.v_biases += (1 - beta2) * (grad_biases**2)

        # # To correct the biases introduced by initializing the first and second moment vectors at 0, Adam computes bias-corrected estimates
        # m_weights_hat = self.m_weights / (1 - beta1**self.time_step)
        # m_biases_hat = self.m_biases / (1 - beta1**self.time_step)

        # v_weights_hat = self.v_weights / (1 - beta2**self.time_step)
        # v_biases_hat = self.v_biases / (1 - beta2**self.time_step)
        pass
