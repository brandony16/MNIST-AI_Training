import cupy as cp


def use_optimizer(parameters, type="Adam", lr=0.001):
    match (type.upper()):
        case "SGD":
            return SGD(parameters, lr)
        case "ADAM":
            return Adam(parameters, learningRate=lr)
        case _:
            return Adam(parameters, learningRate=lr)


class SGD:
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters  # List of (param, grad) tuples
        self.lr = lr

    def step(self):
        for param, grad in self.parameters:
            param -= self.lr * grad

    def zero_grad(self):
        for _, grad in self.parameters:
            grad.fill(0.0)


class Adam:
    def __init__(
        self, parameters, learningRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
    ):
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
