import numpy as np

def softmax(x):
  exp_x = np.exp(x - np.max(x))
  return exp_x / exp_x.sum(axis=0)

class CNN:
  def __init__(self, layers):
    self.layers = layers

  def forward(self, image):
    output = image
    for layer in self.layers:
      output = layer.forward(output)

    return softmax(output)