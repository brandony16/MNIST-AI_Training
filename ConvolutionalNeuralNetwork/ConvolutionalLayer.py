import numpy as np

class ConvLayer:
  def __init__(self, numFilters, filterSize):
    self.numFilters = numFilters
    self.filterSize = filterSize
    self.filters = np.random.randn(numFilters, filterSize, filterSize) / (filterSize ** 2)
  
  def iterateRegions(self, image):
    # Yeilds region of size filterSize x filterSize and its top left coordinates: i and j
    height, width=  image.shape
    for i in range(height - self.filterSize + 1):
      for j in range(width - self.filterSize + 1):
        region = image[i:(i + self.filterSize), j:(j + self.filterSize)]
        yield region, i, j

  def forwardPass(self, input):
    # Store input for backpropogation
    self.lastInput = input
    height, width = input.shape
    output = np.zeros((height - self.filterSize + 1, width - self.filterSize + 1, self.numFilters))

    #  For each region of the input image, we compute the dot product of the region and each filter, storing the result in the output feature map.
    for region, i, j in self.iterateRegions(input):
      output[i,j] = np.sum(region * self.filters, axis=(1, 2))
    
    return output
  
  def backprop(self, dL_dOutput, learnRate):
    # dL_dOutput is the gradient of the loss with respect to the output of this layer.
    dL_dFilters = np.zeros(self.filters.shape)

    for region, i, j in self.iterateRegions(self.lastInput):
      for f in range(self.numFilters):
        dL_dFilters[f] = dL_dOutput[i, j, f] * region

    self.filters -= learnRate * dL_dFilters
    return dL_dFilters