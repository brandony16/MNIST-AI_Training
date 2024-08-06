import numpy as np
from numba import njit, prange

class Dense:
  def __init__(self, input_size, output_size, learning_rate=0.01, activation='relu'):
      self.input_size = input_size
      self.output_size = output_size
      self.learning_rate = learning_rate
      self.activation = activation
      self.weights = np.random.randn(input_size, output_size) * 0.01
      self.biases = np.zeros((1, output_size))
      self.input = None
      self.output = None
      self.activated_output = None

  def forwardPass(self, input):
      self.input = input
      self.output = np.dot(self.input, self.weights) + self.biases
      self.activated_output = self._apply_activation(self.output)
      return self.activated_output

  def backprop(self, d_output, learn_rate):
      if self.activation == 'softmax':
          d_output = self._apply_softmax_derivative(d_output)
      else:
          d_output = self._apply_activation_derivative(d_output)

      d_input, d_weights, d_biases = self._compute_gradients(d_output, self.input, self.weights)
      
      self.weights -= self.learning_rate * d_weights
      self.biases -= self.learning_rate * d_biases

      return d_input

  @staticmethod
  @njit(parallel=True)
  def _compute_gradients(d_output, input, weights):
    d_input = np.dot(d_output, weights.T)
    d_weights = np.dot(input.T, d_output)
    d_biases = np.sum(d_output, axis=0)
    return d_input, d_weights, d_biases

  def _apply_activation(self, x):
    if self.activation == 'relu':
        return np.maximum(0, x)
    elif self.activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    elif self.activation == 'tanh':
        return np.tanh(x)
    elif self.activation == 'softmax':
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return x

  def _apply_activation_derivative(self, d_output):
    if self.activation == 'relu':
        d_output[self.output <= 0] = 0
    elif self.activation == 'sigmoid':
        sigmoid = 1 / (1 + np.exp(-self.output))
        d_output *= sigmoid * (1 - sigmoid)
    elif self.activation == 'tanh':
        d_output *= 1 - np.tanh(self.output) ** 2
    return d_output

  def _apply_softmax_derivative(self, d_output):
      return d_output


# dense_layer = Dense(input_size=3, output_size=2, learning_rate=0.001, activation='relu')

# # Forward pass
# input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# output_data = dense_layer.forwardPass(input_data)
# print("Forward pass output:")
# print(output_data)

# # Backward pass
# d_output = np.array([[1.0, 0.5], [0.2, 0.3]])
# d_input = dense_layer.backprop(d_output)
# print("Backward pass output:")
# print(d_input)

# input_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# output_data = dense_layer.forwardPass(input_data)
# print("Forward pass output:")
# print(output_data)