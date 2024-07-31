import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from BasicNeuralNetwork.neuralNetwork import NeuralNetwork
from loadMNIST import load_and_preprocess_mnist

if __name__ == "__main__":
  print("Starting Program")
  train_images, train_labels, test_images, test_labels = load_and_preprocess_mnist()
  # Define the neural network architecture
  layer_sizes = [784, 248, 10] 

  # relu and sigmoid activation functions available
  # reLU: Best accuracy: 97.45% Layers: 784, 512, 128, 10 Epochs: 10 Learning Rate: 0.01 Mini-batch Size: 32
  nn = NeuralNetwork(layer_sizes, 'relu')
  print("Beginning Training")
  nn.train(train_images, train_labels, 10, 0.1, 32)

  print("Evaluating on test data:")
  test_output = nn.forward(test_images)
  test_predictions = nn.predict(test_images)
  test_loss = nn.cross_entropy(test_labels, test_output)
  print(f'Test Loss: {test_loss}')
  
  # Calculate accuracy
  test_labels_argmax = np.argmax(test_labels, axis=1)
  accuracy = np.mean(test_predictions == test_labels_argmax)
  print(test_predictions[:10], test_labels_argmax[:10])
  print(f'Test Accuracy: {accuracy * 100:.2f}%')

  # Calculate confusion matrix, precision, recall, and F1-score
  cm = confusion_matrix(test_labels_argmax, test_predictions)
  precision = precision_score(test_labels_argmax, test_predictions, average=None)
  recall = recall_score(test_labels_argmax, test_predictions, average=None)
  f1 = f1_score(test_labels_argmax, test_predictions, average=None)
  
  print("Confusion Matrix:")
  print(cm)
  print("Precision per class:")
  print(precision)
  print("Recall per class:")
  print(recall)
  print("F1-score per class:")
  print(f1)