import numpy as np

class ReLU:
    def forwardPass(self, input):
        self.last_input = input
        return np.maximum(0, input)
    
    def backprop(self, dL_dOutput, learnRate):
        dL_dInput = dL_dOutput * (self.last_input > 0)

        return dL_dInput