import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from neuralNetwork import NeuralNetwork

# Load the MNIST data from the npz file
def load_mnist_data(npz_file='mnist.npz'):
    data = np.load(npz_file)
    train_images = data['train_images']
    train_labels = data['train_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    # Normalize the images
    train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
    
    # One-hot encode the labels
    train_labels = np.eye(10)[train_labels]
    test_labels = np.eye(10)[test_labels]
    
    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
  train_images, train_labels, test_images, test_labels = load_mnist_data()
  
  # Define the neural network architecture
  layer_sizes = [784,100, 10] 
  

  nn = NeuralNetwork(layer_sizes, 'sigmoid')
  nn.train(train_images, train_labels, 10, 0.01, 32)

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