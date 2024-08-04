from loadMNIST import load_and_preprocess_mnist
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
from ConvolutionalNeuralNetwork.CNNClass import CNN
from ConvolutionalNeuralNetwork.ConvolutionalLayer import ConvLayer
from ConvolutionalNeuralNetwork.DenseLayer import Dense
from ConvolutionalNeuralNetwork.PoolingLayer import Pooling
from ConvolutionalNeuralNetwork.ReLUClass import ReLU
from ConvolutionalNeuralNetwork.Flatten import flatten

trainImages, trainLabels, testImages, testLabels = load_and_preprocess_mnist()
trainImages = trainImages.reshape(-1, 28, 28)
testImages = testImages.reshape(-1, 28, 28)
print('Data loaded')

architecture = [
    ConvLayer(num_filters=6, filter_size=5),
    ReLU(),
    # Pooling(),
    # ConvLayer(numFilters=16, filterSize=5),
    # ReLU(),
    # Pooling(),
    # ConvLayer(numFilters=120, filterSize=5),
    # ReLU(),
    flatten(),
    # Dense(input_size=576, output_size=15, activation='relu'),
    Dense(input_size=3456, output_size=10, activation='softmax')
]

cnn = CNN(layers=architecture)

print("Training Started")
cnn.train(trainImages, trainLabels, epochs=1, learn_rate=0.005, batch_size=32)

print("Testing Started")
predictedLabels = np.array(cnn.predict(testImages))

test_labels_argmax = np.argmax(testLabels, axis=1)
accuracy = np.mean(predictedLabels == test_labels_argmax)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Calculate confusion matrix, precision, recall, and F1-score
cm = confusion_matrix(test_labels_argmax, predictedLabels)
precision = precision_score(test_labels_argmax, predictedLabels, average=None)
recall = recall_score(test_labels_argmax, predictedLabels, average=None)
f1 = f1_score(test_labels_argmax, predictedLabels, average=None)

print("Confusion Matrix:")
print(cm)
print("Precision per class:")
print(precision)
print("Recall per class:")
print(recall)
print("F1-score per class:")
print(f1)
