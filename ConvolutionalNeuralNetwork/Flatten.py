import numpy as np

class flatten:
    def forwardPass(self, input):
        self.input_shape = input.shape
        return input.flatten().reshape(input.shape[0], -1)

    def backprop(self, d_output, learning_rate):
        return d_output.reshape(self.input_shape)