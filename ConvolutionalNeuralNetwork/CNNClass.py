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
      output = layer.forwardPass(output)

    return softmax(output)
  
  def backprop(self, d_output, learning_rate):
    for layer in reversed(self.layers):
      d_output = layer.backprop(d_output, learning_rate)
    return d_output
  
  def train(self, images, labels, epochs=1, learn_rate=0.005, batch_size=32):
    num_samples = images.shape[0]
    num_batches = num_samples // batch_size

    for epoch in range(epochs):
      epoch_loss = 0
      for batch in range(num_batches):
          start = batch * batch_size
          end = start + batch_size
          x_batch = images[start:end]
          y_batch = labels[start:end]

          # Forward pass
          predictions = self.forward(x_batch)

          # Compute loss (cross-entropy loss)
          loss = self._cross_entropy_loss(predictions, y_batch)
          epoch_loss += loss

          # Compute gradient of the loss with respect to predictions
          d_output = self._cross_entropy_loss_derivative(predictions, y_batch)

          # Backward pass
          self.backprop(d_output, learn_rate)

      epoch_loss /= num_batches
      print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')

  def _cross_entropy_loss(self, predictions, labels):
      n_samples = labels.shape[0]
      clipped_preds = np.clip(predictions, 1e-12, 1 - 1e-12)
      correct_confidences = np.sum(labels * np.log(clipped_preds), axis=1)
      loss = -np.mean(correct_confidences)
      return loss

  def _cross_entropy_loss_derivative(self, predictions, labels):
      n_samples = labels.shape[0]
      clipped_preds = np.clip(predictions, 1e-12, 1 - 1e-12)
      return (clipped_preds - labels) / n_samples
  
  def predict(self, image):
        out = self.forward(image)
        return np.argmax(out)
