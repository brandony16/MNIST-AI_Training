import numpy as np

def softmax(x):
  exp_x = np.exp(x - np.max(x))
  return exp_x / exp_x.sum(axis=0)

def categoricalCrossEntropy(predictions, labels):
  # Ensure predictions are clipped to avoid log(0)
  predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
  # Compute categorical cross entropy
  loss = -np.sum(labels * np.log(predictions))
  return loss

class CNN:
  def __init__(self, layers):
    self.layers = layers

  def forward(self, image):
    output = image
    for layer in self.layers:
      output = layer.forwardPass(output)

    return softmax(output)
  
  def train(self, images, labels, epochs=1, learn_rate=0.005, batch_size=32):
    for epoch in range(epochs):
      loss = 0
      permutation = np.random.permutation(len(images))
      images = images[permutation]
      labels = labels[permutation]

      for i in range(0, len(images), batch_size):
        batchImages = images[i: i+batch_size]
        batchLabels = labels[i: i+batch_size]

        # Training on single batch
        batchSizeActual = len(batchImages)
        batchLoss = 0

        # Forward pass
        outputs = []
        for j in range(batchSizeActual):
          out = self.forward(batchImages[j])
          outputs.append(out)
          batchLoss += categoricalCrossEntropy(out, batchLabels[j])
        batchLoss /= batchSizeActual
        loss += batchLoss

        gradient = []
        for j in range(batchSizeActual):
          gradient.append(outputs[j] - batchLabels[j])
        
        for j in range(batchSizeActual):
          grad = gradient[j]
          for layer in reversed(self.layers):
            grad = layer.backprop(grad, learn_rate)

        print(f'Epoch {epoch+1}, Loss: {loss/(len(images)/batch_size)}')

  def predict(self, image):
        out = self.forward(image)
        return np.argmax(out)
