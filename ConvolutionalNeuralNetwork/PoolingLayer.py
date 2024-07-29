import numpy as np

class Pooling:
  # Generates the different regions of the input image over which pooling will be applied
  def iterateRegions(self, input):
    height, width, _ = input.shape
    newHeight = height // 2
    newWidth = width // 2

    for i in range(newHeight):
      for j in range(newWidth):
        region = input[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
        yield region, i, j
  
  def forwardPass(self, input):
    self.lastInput = input
    h, w, numFilters = input.shape
    output = np.zeros((h//2, w//2, numFilters))

    for region, i, j in self.iterateRegions(input):
      # Find max value in each filter
      output[i,j] = np.amax(region, axis=(0,1))