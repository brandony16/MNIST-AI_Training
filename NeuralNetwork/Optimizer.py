import cupy as cp


def use_optimizer(parameters, type="Adam", lr=0.001):
    """
    Gets the correct optimizer.

    Args:
        parameters: list of (param, grad) tuples, where
                         - param is a cupy.ndarray of weights/biases
                         - grad  is a cupy.ndarray of the same shape holding its gradient
        type: the type of optimizer to use (SGD or Adam)
        lr: learning rate
    """
    match (type.upper()):
        case "SGD":
            return SGD(parameters, lr)
        case "ADAM":
            return Adam(parameters, learningRate=lr)
        case _:
            return Adam(parameters, learningRate=lr)


class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.8, decay=1e-4):
        """
        parameters: list of (param, grad) tuples, where
                         - param is a cupy.ndarray of weights/biases
                         - grad  is a cupy.ndarray of the same shape holding its gradient
        lr: learning rate
        momentum: momentum factor (0 for vanilla SGD)
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

        # Initialize one velocity buffer per param tensor
        self.velocities = [cp.zeros_like(param) for param, _ in self.parameters]

    def zero_grad(self):
        """Zero out all gradients."""
        for _, grad in self.parameters:
            grad.fill(0.0)

    def step(self):
        """Perform a single SGD+momentum update."""
        for (param, grad), v in zip(self.parameters, self.velocities):
            param *= 1 - self.lr * self.decay

            # v = momentum * v - lr * grad
            v = self.momentum * v - self.lr * grad

            param += v

    def set_lr(self, lr):
        self.lr = lr


class Adam:
    def __init__(
        self, parameters, learningRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
        """
        parameters: list of (param, grad) tuples, where
                         - param is a cupy.ndarray of weights/biases
                         - grad  is a cupy.ndarray of the same shape holding its gradient
        learningRate: learning rate
        """
        self.parameters = parameters  # List of (param, grad) tuples

        self.lr = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = [cp.zeros_like(p[0]) for p in parameters]
        self.v = [cp.zeros_like(p[0]) for p in parameters]

    def step(self):
        self.t += 1
        for i, (param, grad) in enumerate(self.parameters):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad**2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Parameter update
            param -= self.lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        for _, grad in self.parameters:
            grad.fill(0.0)

    def set_lr(self, lr):
        self.lr = lr
