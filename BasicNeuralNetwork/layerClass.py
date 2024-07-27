import numpy as np

class Layer:
  def __init__(self, inputSize, outputSize, activation='relu'):
    np.random.seed(42)
    # Initializes a matrix of dimensons inputSize x outputSize of weights
    self.weights = np.random.randn(inputSize, outputSize) * np.sqrt(2. / inputSize)
    # Inititalizes an array of biases for each 'neuron'
    self.bias = np.zeros(outputSize)
    self.activation = activation

  
  def forward(self, inputs):
    self.inputs = inputs
    self.linear_output = np.dot(inputs, self.weights) + self.bias
    if self.activation == 'relu':
      self.output = self.reLU(self.linear_output)
    elif self.activation == 'softmax':
      self.output = self.softmax(self.linear_output)    
    return self.output
  
  def backward(self, error, learningRate=0.001):
    # The gradient of the loss with respect to the pre-activation input of the current layer.
    if self.activation == 'relu':
      delta = error * self.reLU_deriv(self.output)
    elif self.activation == 'softmax':
      delta = error # for softmax, error is already the derivative

    # Update weights with dot product of transposed inputs and delta. Each element represents the gradient of the loss with respect to the corresponding weight
    self.weights -= learningRate * np.dot(self.inputs.T, delta)
    # Update biases. Sum gives us a vector where each element represents the gradient of the loss with respect to the corresponding bias.
    self.bias -= learningRate * np.sum(delta, axis=0)
    # Gives error to be backpropogated to previous layer
    return np.dot(delta, self.weights.T)

  def reLU(self, x):
    return np.maximum(0, x)

  def reLU_deriv(self, x):
    return np.where(x > 0, 1, 0)
  
  def softmax(self, x):
    exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)


