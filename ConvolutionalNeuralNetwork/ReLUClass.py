import numpy as np

class ReLU:
    def __init__(self):
        self.input = None

    def forwardPass(self, input):
        self.input = input
        return np.maximum(0, input)
    
    def backprop(self, d_output, learn_rate):
        d_input = d_output.copy()
        d_input[self.input <= 0] = 0

        return d_input