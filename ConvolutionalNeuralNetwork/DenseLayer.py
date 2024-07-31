import numpy as np

class Dense:
  def __init__(self, inputSize, outputSize, activation='relu'):
    np.random.seed(42)
    # Initializes a matrix of dimensons inputSize x outputSize of weights
    self.weights = np.random.randn(inputSize, outputSize) * np.sqrt(2. / inputSize)
    # Inititalizes an array of biases for each 'neuron'
    self.bias = np.zeros(outputSize)
    self.activation = activation

    # Adam Optimization Vars
    self.m_weights = np.zeros_like(self.weights) # First moment vector m. Moving average of past gradients
    self.v_weights = np.zeros_like(self.weights) # Second moment vecter v. Moving average of the squared gradients
    self.m_biases = np.zeros_like(self.bias)
    self.v_biases = np.zeros_like(self.bias)
    self.time_step = 0 # Used to calculate gradient gt at current time step t

  
  def forwardPass(self, inputs):
    self.inputs = inputs
    self.linear_output = np.dot(inputs, self.weights) + self.bias
    if self.activation == 'relu':
      self.output = self.reLU(self.linear_output)
    elif self.activation == 'sigmoid':
      self.output = self.sigmoid(self.linear_output)
    elif self.activation == 'softmax':
      self.output = self.softmax(self.linear_output)    
    return self.output
  
  def backprop(self, error, learningRate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # The gradient of the loss with respect to the pre-activation input of the current layer.
    if self.activation == 'relu':
      delta = error * self.reLU_deriv(self.output)
    elif self.activation == 'sigmoid':
      delta = error * self.sigmoid_derivative(self.output)
    elif self.activation == 'softmax':
      delta = error # for softmax, error is already the derivative

    grad_weights = np.dot(self.inputs.T, delta)
    grad_biases = np.sum(delta, axis=0)

    self.time_step += 1
    # First moment vector. Beta is decay rate. Higher beta gives more weight toward previous gradients
    self.m_weights = beta1 * self.m_weights + (1 - beta1) * grad_weights
    self.m_biases = beta1 * self.m_biases + (1 - beta1) * grad_biases
    # Second moment vector.
    self.v_weights = beta2 * self.v_weights + (1 - beta2) * (grad_weights ** 2)
    self.v_biases = beta2 * self.v_biases + (1 - beta2) * (grad_biases ** 2)

    # To correct the biases introduced by initializing the first and second moment vectors at 0, Adam computes bias-corrected estimates
    m_weights_hat = self.m_weights / (1 - beta1 ** self.time_step)
    m_biases_hat = self.m_biases / (1 - beta1 ** self.time_step)

    v_weights_hat = self.v_weights / (1 - beta2 ** self.time_step)
    v_biases_hat = self.v_biases / (1 - beta2 ** self.time_step)


    # Update weights and biases using formula. Epsilon prevents divison by 0 
    self.weights -= learningRate * m_weights_hat / (np.sqrt(v_weights_hat) + epsilon)

    self.bias -= learningRate * m_biases_hat / (np.sqrt(v_biases_hat) + epsilon)

    # Gives error to be backpropogated to previous layer
    return np.dot(delta, self.weights.T)

  def reLU(self, x):
    return np.maximum(0, x)

  def reLU_deriv(self, x):
    return np.where(x > 0, 1, 0)
  
  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def sigmoid_derivative(self, x):
    s = self.sigmoid(x)
    return s * (1 - s)
  
  def softmax(self, x):
    exp_vals = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)