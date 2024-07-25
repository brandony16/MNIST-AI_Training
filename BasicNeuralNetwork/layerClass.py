import numpy as np

class Layer:
  def __init__(self, inputSize, outputSize):
    np.random.seed(42)
    # Initializes a matrix of dimensons inputSize x outputSize of weights
    self.weights = np.random.rand(inputSize, outputSize)
    # Inititalizes an array of biases for each 'neuron'
    self.bias = np.random.rand(outputSize)
  
  def forward(self, inputs):
    self.inputs = inputs
    self.output = self.reLU(np.dot(inputs, self.weights) + self.bias)
    return self.output
  
  def backward(self, error, learningRate=0.1):
    # The gradient of the loss with respect to the pre-activation input of the current layer.
    delta = error * self.reLU_deriv(self.output)
    # Update weights with dot product of transposed inputs and delta. Each element represents the gradient of the loss with respect to the corresponding weight
    self.weights += learningRate * np.dot(self.inputs.T, delta)
    # Update biases. Sum gives us a vector where each element represents the gradient of the loss with respect to the corresponding bias.
    self.bias += learningRate * np.sum(delta, axis=0)
    # Gives error to be backpropogated to previous layer
    return np.dot(delta, self.weights.T)

  def reLU(self, x):
    return np.maximum(0, x)

  def reLU_deriv(self, x):
    return np.where(x > 0, 1, 0)

