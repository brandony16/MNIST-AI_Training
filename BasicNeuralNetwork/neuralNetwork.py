import numpy as np
from BasicNeuralNetwork.layerClass import Layer

class NeuralNetwork:
  def __init__(self, layer_sizes, activation='relu'):
    self.layers = []
    for i in range(len(layer_sizes) - 1):
      activation = activation if i < len(layer_sizes) - 2 else 'softmax'
      self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activation))
  
  def forward(self, X):
    for layer in self.layers:
      X = layer.forward(X)
    return X

  def backward(self, X, y, output, learningRate):
    error = self.cross_entropy_derivative(y, output)
    for layer in reversed(self.layers):
      error = layer.backward(error, learningRate)

  def train(self, data, labels, epochs=10, learningRate=0.1, batch_size=32):
    num_samples = data.shape[0]
    for epoch in range(epochs):
      # Shuffle the data at the beginning of each epoch
      indices = np.arange(num_samples)
      np.random.shuffle(indices)
      data = data[indices]
      labels = labels[indices]
      print("Epoch Started")

      # Process the data in batches
      for start_idx in range(0, num_samples, batch_size):
          end_idx = min(start_idx + batch_size, num_samples)
          batch_data = data[start_idx:end_idx]
          batch_labels = labels[start_idx:end_idx]
          
          # Forward and backward pass for the batch
          output = self.forward(batch_data)
          self.backward(batch_data, batch_labels, output, learningRate)

      if epoch % epochs//10 == 0:
        print(f"Epoch {epoch} of {epochs} completed")
        print(f"Training Loss: {self.cross_entropy(labels, self.forward(data))}")

  # Loss calculation
  def cross_entropy(self, y, output):
    # Clip values to prevent log(0)
    output = np.clip(output, 1e-12, 1.0 - 1e-12)
    return -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))

  def cross_entropy_derivative(self, y, output):
    return output - y
  
  def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
  
  
# Example usage
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    # Define layer sizes: 2 input neurons, 1 hidden layers with 3 neurons, and 2 output neurons
    layer_sizes = [2, 10, 5, 3, 2]

    nn = NeuralNetwork(layer_sizes)
    nn.train(X, y, 1000, 0.01)
    print("Predicted Output:\n", nn.predict(X))