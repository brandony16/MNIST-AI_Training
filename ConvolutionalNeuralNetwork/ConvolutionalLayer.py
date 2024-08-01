import numpy as np

class ConvLayer:
  def __init__(self, num_filters, filter_size, stride = 1, padding='valid'):
    self.num_filters = num_filters
    self.filter_size = filter_size
    self.stride = stride
    self.padding = padding
    # Init filters with random values
    self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size ** 2)

  # Pads image
  def _pad_input(self, image):
    if self.padding == 'same':
      pad_height = (self.stride * (input.shape[0] - 1) + self.filter_size - input.shape[0]) // 2
      pad_width = (self.stride * (input.shape[1] - 1) + self.filter_size - input.shape[0]) // 2
      padded_input = np.pad(input, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    elif self.padding == 'valid':
      padded_input = image
    return padded_input

  # Does convolution
  def _convolve(self, input, filter):
    input_height, input_width = input.shape
    filter_height, filter_width = filter.shape

    output_height = (input_height - filter_height) // self.stride + 1
    output_width = (input_width - filter_width) // self.stride + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
      for j in range(output_width):
        h_start = i * self.stride
        h_end = h_start + filter_height
        w_start = j * self.stride
        w_end = w_start + filter_width
        output[i, j] = np.sum(input[h_start:h_end, w_start:w_end] * filter)

    return output
  
  def forward(self, input):
    self.input = self._pad_input(input)
    self.output = np.zeros((self.num_filters, 
                        (self.input.shape[0] - self.filter_size) // self.stride + 1, # Height of output feature map
                        (self.input.shape[1] - self.filter_size) // self.stride + 1)) # WIdth of output feature map
    for f in range(self.num_filters):
      self.output[f] = self._convolve(input, self.filters[f])

    return self.output

input = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
    [3, 2, 1, 0]
])

filter = np.array([
    [1, 0],
    [0, -1]
])

filters = [
    np.array([
        [1, 0],
        [0, -1]
    ]),
    np.array([
        [0, 1],
        [-1, 0]
    ])
]


testLayer = ConvLayer(num_filters=5, filter_size=2)

convolvedMatrix = testLayer._convolve(input, filter)
forwardOutput = testLayer.forward(input)

print("Convolved Matrix:")
print(convolvedMatrix)
print("Forward Output:")
print(forwardOutput)