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
      
    return output

  def backprop(self, dL_dOutput, learnRate):
    dL_dInput = np.zeros(self.lastInput.shape)

    for region, i, j in self.iterateRegions(self.lastInput):
      h, w, f = region.shape
      amax = np.amax(region, axis=(0,1))

      for i2 in range(h):
        for j2 in range(w):
          for k2 in range(f):
            if region[i2, j2, k2] == amax(k2):
              dL_dInput[i * 2 + i2, j * 2 + j2, k2] = dL_dOutput[i, j, k2]
    return dL_dInput