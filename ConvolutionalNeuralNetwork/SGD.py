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
