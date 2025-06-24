import cupy as cp


class ReLU:
    def forward(self, x):
        self.input = x
        return cp.maximum(0, x)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input


class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + cp.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)


class Tanh:
    def forward(self, x):
        self.output = cp.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output**2)


class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.input = x
        return cp.where(x > 0, x, self.alpha * x)

    def backward(self, grad_output):
        return grad_output * cp.where(self.input > 0, 1, self.alpha)