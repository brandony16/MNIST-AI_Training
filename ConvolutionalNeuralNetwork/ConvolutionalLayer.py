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
        pad_height = (self.stride * (inputs.shape[1] - 1) + self.filter_size - inputs.shape[1]) // 2
        pad_width = (self.stride * (inputs.shape[2] - 1) + self.filter_size - inputs.shape[2]) // 2
        padded_input = np.pad(inputs, ((0, 0), (pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    elif self.padding == 'valid':
        padded_input = inputs
    return padded_input

  # Does convolution
  @staticmethod
  @njit(parallel=True)
  def _convolve(input, filters, stride):
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

  
  def forwardPass(self, inputs):
    self.inputs = inputs
    batch_size, input_height, input_width = inputs.shape
    inputs_padded = self._pad_input(inputs)

    if self.padding == 'same':
        output_height = input_height
        output_width = input_width
    else:  # 'valid' padding
        output_height = (input_height - self.filter_size) // self.stride + 1
        output_width = (input_width - self.filter_size) // self.stride + 1

    output = np.zeros((batch_size, self.num_filters, output_height, output_width))

    for b in range(batch_size):
        output[b] = self._convolve(inputs_padded[b], self.filters, self.stride)

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

                region = input_padded[h_start:h_end, w_start:w_end]
                d_filters[f] += region * d_output[f, i, j]
                d_input_padded[h_start:h_end, w_start:w_end] += filters[f] * d_output[f, i, j]

    return d_filters, d_input_padded

  def backprop(self, d_output, learning_rate):
      batch_size, input_height, input_width = self.inputs.shape
      input_padded = self._pad_input(self.inputs)

      d_filters = np.zeros_like(self.filters, dtype=np.float64)
      d_input = np.zeros_like(self.inputs, dtype=np.float64)

      for b in range(batch_size):
          d_f, d_input_padded = self._backprop(d_output[b], input_padded[b], self.filters, self.filter_size, self.stride)
          d_filters += d_f
          d_input[b] = d_input_padded[:input_height, :input_width]  # Assuming padding='same'

      self.filters -= learning_rate * d_filters

      return d_input

# input = np.array([
#     [1, 2, 3, 0],
#     [4, 5, 6, 1],
#     [7, 8, 9, 2],
#     [3, 2, 1, 0]
# ])

# filter = np.array([
#     [1, 0],
#     [0, -1]
# ])

# d_output = np.array([
#     [
#         [0.1, 0.2, 0.3],
#         [0.4, 0.5, 0.6],
#         [0.7, 0.8, 0.9]
#     ],
#     [
#         [0.9, 0.8, 0.7],
#         [0.6, 0.5, 0.4],
#         [0.3, 0.2, 0.1]
#     ]
# ])


# testLayer = ConvLayer(num_filters=2, filter_size=2)

# convolvedMatrix = testLayer._convolve(input, filter)
# forwardOutput = testLayer.forwardPass(input)
# backpropOutput = testLayer.backprop(d_output=d_output, learning_rate=1)
# newForward = testLayer.forwardPass(input)

# print("Convolved Matrix:")
# print(convolvedMatrix)
# print("Forward Output:")
# print(forwardOutput)
# print("Backpropogation:")
# print(backpropOutput)
# print("New Forward Pass")
# print(newForward)
