import numpy as np
from numpy.lib.stride_tricks import as_strided
from numba import njit, prange

class ConvLayer:
  def __init__(self, num_filters, filter_size, stride = 1, padding='valid'):
    self.num_filters = num_filters
    self.filter_size = filter_size
    self.stride = stride
    self.padding = padding
    # Init filters with random values
    self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size ** 2)

  # Pads image
  def _pad_input(self, inputs):
    if self.padding == 'same':
        pad_height = (self.stride * (inputs.shape[-2] - 1) + self.filter_size - inputs.shape[-2]) // 2
        pad_width = (self.stride * (inputs.shape[-1] - 1) + self.filter_size - inputs.shape[-1]) // 2
        if len(inputs.shape) == 4:  # Input with channel dimension
            padded_input = np.pad(inputs, ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='constant')
        else:  # Input without channel dimension
            padded_input = np.pad(inputs, ((0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    elif self.padding == 'valid':
        padded_input = inputs
    return padded_input

  # Does convolution
  @staticmethod
  # @njit(parallel=True)
  def _convolve_2d(input, filters, stride):
      num_filters, filter_height, filter_width = filters.shape
      input_height, input_width = input.shape
      output_height = (input_height - filter_height) // stride + 1
      output_width = (input_width - filter_width) // stride + 1
      output = np.zeros((num_filters, output_height, output_width))

      for f in prange(num_filters):
          for i in range(output_height):
              for j in range(output_width):
                  h_start = i * stride
                  h_end = h_start + filter_height
                  w_start = j * stride
                  w_end = w_start + filter_width
                  output[f, i, j] = np.sum(input[h_start:h_end, w_start:w_end] * filters[f])

      return output

  @staticmethod
  # @njit(parallel=True)
  def _convolve_3d(input, filters, stride):
      num_filters, filter_height, filter_width = filters.shape
      input_depth, input_height, input_width = input.shape
      output_height = (input_height - filter_height) // stride + 1
      output_width = (input_width - filter_width) // stride + 1
      output = np.zeros((num_filters, output_height, output_width))

      for f in prange(num_filters):
          for i in range(output_height):
              for j in range(output_width):
                  h_start = i * stride
                  h_end = h_start + filter_height
                  w_start = j * stride
                  w_end = w_start + filter_width
                  region = input[:, h_start:h_end, w_start:w_end]
                  output[f, i, j] = np.sum(region * filters[f])

      return output

  def forwardPass(self, inputs):
      if len(inputs.shape) == 3:
          inputs = inputs[:, np.newaxis, :, :]  # Add a channel dimension if missing
      self.inputs = inputs
      batch_size, input_depth, input_height, input_width = inputs.shape
      inputs_padded = self._pad_input(inputs)

      if self.padding == 'same':
          output_height = input_height
          output_width = input_width
      else:  # 'valid' padding
          output_height = (input_height - self.filter_size) // self.stride + 1
          output_width = (input_width - self.filter_size) // self.stride + 1

      output = np.zeros((batch_size, self.num_filters, output_height, output_width))

      for b in range(batch_size):
          if input_depth == 1:
              output[b] = self._convolve_2d(inputs_padded[b, 0], self.filters, self.stride)
          else:
              output[b] = self._convolve_3d(inputs_padded[b], self.filters, self.stride)
      
      return output

  @staticmethod
  @njit(parallel=True)
  def _backprop(d_output, input_padded, filters, filter_size, stride):
      num_filters, output_height, output_width = d_output.shape
      d_filters = np.zeros_like(filters, dtype=np.float64)
      d_input_padded = np.zeros_like(input_padded, dtype=np.float64)

      for f in prange(num_filters):
          for i in range(output_height):
              for j in range(output_width):
                  h_start = i * stride
                  h_end = h_start + filter_size
                  w_start = j * stride
                  w_end = w_start + filter_size

                  region = input_padded[:, h_start:h_end, w_start:w_end]
                  d_filters[f] += region * d_output[f, i, j]
                  d_input_padded[:, h_start:h_end, w_start:w_end] += filters[f] * d_output[f, i, j]

      return d_filters, d_input_padded

  def backprop(self, d_output, learning_rate):
      batch_size, input_depth, input_height, input_width = self.inputs.shape
      input_padded = self._pad_input(self.inputs)

      d_filters = np.zeros_like(self.filters, dtype=np.float64)
      d_input = np.zeros_like(self.inputs, dtype=np.float64)

      for b in range(batch_size):
          d_f, d_input_padded = self._backprop(d_output[b], input_padded[b], self.filters, self.filter_size, self.stride)
          d_filters += d_f
          d_input[b] = d_input_padded[:, :input_height, :input_width]  # Assuming padding='same'

      self.filters -= learning_rate * d_filters

      return d_input

# # Example usage
# conv_layer = ConvLayer(num_filters=6, filter_size=3, stride=1, padding='same')

# # Initial input without channel dimension
# input_tensor = np.random.rand(32, 28, 28)
# conv_output = conv_layer.forwardPass(input_tensor)

# # For backpropagation
# d_output = np.random.rand(32, 6, 28, 28)  # Example gradient of the loss with respect to the conv output
# d_input = conv_layer.backprop(d_output, learning_rate=0.01)

# print("Forward Ouput:")
# print(conv_output)
# print("Backpropogation:")
# print(d_input)

