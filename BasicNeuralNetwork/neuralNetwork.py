import numpy as np
from layerClass import Layer

class NeuralNetwork:
  def __init__(self, layers):
    self.layers = []
    for i in range(len(layers) - 1):
      self.layers.append(Layer(layers[i], layers[i+1]))
  
  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def backward(self, X, y, output):
    error = y - output
    for layer in reversed(self.layers):
      error = layer.backward(error)

  def train(self, data, labels, epochs=10000, learningRate=0.1):
    for epoch in range(epochs):
      output = self.forward(data)
      self.backward(data, labels, output)
      if epoch % 1000 == 0:
        loss = np.mean(np.square(y - output))
        print(f'Epoch {epoch}, Loss: {loss}')
  
# Example usage
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Define layer sizes: 2 input neurons, 1 hidden layers with 3 neurons, and 2 output neurons
    layer_sizes = [2, 100, 2]

    nn = NeuralNetwork(layer_sizes)
    nn.train(X, y)
    print("Predicted Output:\n", nn.forward(X))