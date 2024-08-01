import numpy as np

class flatten:
    def forwardPass(self, input):
        return input.reshape(input.shape[0], -1)
    
    def backprop(self, d_output):
        return d_output