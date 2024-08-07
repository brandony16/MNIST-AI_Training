import numpy as np
from numba import njit, prange

class Pooling:
  def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.output = None
        self.argmax = None

  def forwardPass(self, inputs):
      self.input = inputs
      if len(inputs.shape) == 3:
          inputs = inputs[:, np.newaxis, :, :]  # Add a channel dimension if missing

      batch_size, num_channels, input_height, input_width = inputs.shape
      
      # Calculate output dimensions
      output_height = (input_height - self.pool_size) // self.stride + 1
      output_width = (input_width - self.pool_size) // self.stride + 1
      self.output = np.zeros((batch_size, num_channels, output_height, output_width))
      self.argmax = np.zeros_like(self.output, dtype=np.int32)

      self.output, self.argmax = self._forward(inputs, self.pool_size, self.stride, self.output, self.argmax)
      return self.output

  def backprop(self, d_output):
      if len(self.input.shape) == 3:
          self.input = self.input[:, np.newaxis, :, :]  # Add a channel dimension if missing
      d_input = np.zeros_like(self.input)
      d_input = self._backward(d_output, self.input, self.argmax, self.pool_size, self.stride, d_input)
      if d_input.shape[1] == 1:
          d_input = d_input[:, 0, :, :]  # Remove the channel dimension if it was added
      return d_input

  @staticmethod
  @njit(parallel=True)
  def _forward(inputs, pool_size, stride, output, argmax):
      batch_size, num_channels, input_height, input_width = inputs.shape
      output_height = (input_height - pool_size) // stride + 1
      output_width = (input_width - pool_size) // stride + 1

      for b in prange(batch_size):
          for c in range(num_channels):
              for i in range(output_height):
                  for j in range(output_width):
                      h_start = i * stride
                      h_end = h_start + pool_size
                      w_start = j * stride
                      w_end = w_start + pool_size
                      
                      region = inputs[b, c, h_start:h_end, w_start:w_end]
                      output[b, c, i, j] = np.max(region)
                      argmax[b, c, i, j] = np.argmax(region)

      return output, argmax

  @staticmethod
  @njit(parallel=True)
  def _backward(d_output, inputs, argmax, pool_size, stride, d_input):
      batch_size, num_channels, output_height, output_width = d_output.shape

      for b in prange(batch_size):
          for c in range(num_channels):
              for i in range(output_height):
                  for j in range(output_width):
                      h_start = i * stride
                      h_end = h_start + pool_size
                      w_start = j * stride
                      w_end = w_start + pool_size

                      region = inputs[b, c, h_start:h_end, w_start:w_end]
                      argmax_value = argmax[b, c, i, j]
                      (h_index, w_index) = np.unravel_index(argmax_value, region.shape)

                      d_input[b, c, h_start + h_index, w_start + w_index] += d_output[b, c, i, j]

      return d_input